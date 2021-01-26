import pandas as pd
import numpy as np
from pathlib import Path

import torch as th
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from . import GraphEmbeddings
from preprocessing.Document import Document
from loguru import logger

# from .losses import binary_focal_loss

"""
Entity linking :

Candidate Generation


Mention Disambiguation Features:

- dot(v_doc, v_en)
- the entity popularity of e,
- the prior probability of m referring to e,
- and the maximum prior probability of e of all mentions in t.

optionally :
- string similarities(title of e, the surface of m)
- title == m ?
- title.startswith(m) or title.endswith(m)
- title of e starts or ends with the m.


Pour chaque entité de la base de connaissance,
on détermine l'ensemble des contextes de cette entité
- contextes globaux de l'entité : au niveau document
- contextes locaux : au niveau paragraphe

Ces contextes à un vocabulaire que l'on peut sommer/moyenner/concatener pour générer des entity embeddings
Ces entity embeddings seront ensuite utilisés par un classifieur qui va concatener

- document embeddings
- paragraph embeddings
- entity token embeddings (gère la morphologie de l'entité)
- entity embeddings (local)
- entity embeddings (global)
- entity from knowledge embeddings (structurel)
- entity type


- dot(v_doc, v_en)
- the entity popularity of e,
- the prior probability of m referring to e,
- and the maximum prior probability of e of all mentions in t.

Features
- mentions aux entités locales vs mentions aux entiés globales
- mentions aux entités locales vs mentions aux entiés globales
- mentions aux entités locales vs mentions aux entiés globales
- mentions aux entités locales vs document
- mentions aux entités globales vs document

Y = [e0+, e01-, e0m-, e] : vecteur binaire qui indique pour chaque entité si elle appartient au corpus ou non
"""

# biblio : https://github.com/NetherNova/event-kge/ TransE + Skipgram trained jointly
# https://github.com/chihming/awesome-network-embedding


class EntityLinker(LightningModule):
    """Gestion des cas d'ambiguités repérée par l'annotateur par un classifieur binaire Keras"""

    def __init__(self, linker_name, path, wsd=True, dim=50):
        # TODO : éviter de tokenizer le texte alors que ça à déjà été fait

        super().__init__()
        self.save_hyperparameters()

        self.name = linker_name
        self.path = Path(path).expanduser() / linker_name
        self.model_path = self.path / "model.pth"

        if not self.path.exists():
            self.path.mkdir(parents=True)

        if self.name == "entity_linking":
            self.entity_embeddings = TransEmbeddings(self.path)
            self.id_label = "entiteId"
            self.score_label = "score"

        elif self.name == "bdc2wiki":
            emb_path = (
                "/home/tgirault/data/reco/pretrained/wikifier2019/bdc2wiki/kv_trans"
            )
            self.entity_embeddings = WikiEmbeddings(emb_path)
            self.id_label = "entiteId"
            self.score_label = "score"

        elif self.name == "wikifier":
            self.entity_embeddings = WikiEmbeddings()
            self.id_label = "wikiId"
            self.score_label = "scoreWiki"

        self.model = None
        self.wsd = wsd

        if self.model_path.exists():
            self.load_model()
        else:
            logger.warning(f"EntityLinker : modèle {self.path} inexistant")
            self.wsd = False

    def fit_embeddings(self):
        self.entity_embeddings.fit()

    def forward(self, x):
        return self.net(x.float())

    def prepare_doc(self, extracted_entities, en_positive, doc_vec):
        """Transforme un document et sa liste d'entités positives en un vecteur classifiable"""
        if extracted_entities.empty:
            return
        ens_seq, ens = self.group_candidates(extracted_entities)
        mean_rels, vecs, sim, ens_in_rels = self.entity_embeddings.get_spectrum(
            ens, ens_seq
        )
        if not ens_in_rels:
            return

        # TODO : On doit pouvoir tout faire avec une beau produit matriciel
        for en_seq in ens_seq:
            en_seq = en_seq.split()
            l = [
                (i, en in en_positive)
                for i, en in enumerate(ens_in_rels)
                if en in en_seq
            ]
            if l:
                en_idx, y = zip(*l)
                if any(y):
                    for i, label in l:
                        yield ens_in_rels[i], th.hstack(
                            (mean_rels, vecs[i], sim[i], doc_vec)
                        ), label
                elif not en_positive:
                    for i, label in l:
                        yield ens_in_rels[i], th.hstack(
                            (mean_rels, vecs[i], sim[i], doc_vec)
                        ), []

    def group_candidates(self, df):
        """Transforme un ensemble d'entités extraites en une liste d'entités"""
        # TODO : transformation en matrice pivot avec (rows, cols) = (ens_seq, ens)
        if self.id_label not in df.columns:
            return [], []
        df[self.id_label] = df[self.id_label].fillna("")
        ens_seq = (
            df.groupby(by=["start", "end"])[self.id_label]
            .apply(lambda l: " ".join(l))
            .reset_index()
        )
        ens_seq = ens_seq[self.id_label].unique().tolist()
        ens = df[self.id_label].unique().tolist()
        return ens_seq, ens

    def score_entities(self, extractions, doc_vec):
        """Classement des entités nommées à partir d'un modèle entrainé sur corpus d'apprentissage"""
        x = tuple(zip(*self.prepare_doc(extractions, [], doc_vec)))
        if x:
            en_list, m_en_doc, _ = x
            preds = self.predict(th.tensor(m_en_doc))
            return pd.DataFrame({self.id_label: en_list, "score": preds[:, 0]})
        else:
            return pd.DataFrame({self.id_label: [], self.score_label: []})
        # pred =  np.around(, decimals=3)
        # scores["score"] = scores["score"].round(decimals=3)

    def get_entities_dataframe(self, doc: Document, doc_vec=None):
        """Extrait d'un texte une liste d'entités et détermine leur score de pertinence"""
        entities = doc.entities
        if entities.shape[0] == 0:
            return pd.DataFrame([])
        if self.wsd:
            doc_vec = doc_vec if doc_vec is not None else doc.vector
            scores = self.score_entities(entities, doc_vec)
            entities = entities.merge(scores, on=self.id_label, how="outer")
            # entities.columns = ['commonness', 'end', 'entityId', 'id', 'start', 'entiteForme', 'score']
            entities.fillna(-1, inplace=True)

            # si le nombre de formes est > 1, il y a suffisament de contexte pour désambiguïser
            if len(entities["entiteForme"].unique()) > 1:
                best_scores = entities.groupby(["start", "end"])["score"].transform(max)
                entities = entities[entities["score"] == best_scores]
                # entities = entities[entities.score.abs() > 0.15]
        else:
            entities["score"] = -1
        return entities

    def get_average_vector(self, doc):
        try:
            df_ents = self.get_entities_dataframe(doc)
            return self.entity_embeddings.vectors[df_ents[self.id_label]].mean(axis=0)
        except Exception as e:
            return th.zeros(self.entity_embeddings.dim)

    def init_net(self, input_dim, hidden_size=512):
        """Modèle Pytorch pour calculer les exemples positifs et négatifs pour chaque ensemble d'entités
        entrée = ([v_r : contexte référentiel, v_e : entité, similarité(v_r, v_e)]
        sortie = 1 ou 0 si l'entité 0 est positive ou négative dans le contexte référentiel donné
        """
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
        )
        self.bce_loss = nn.BCELoss()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-2)

    def calculate_loss(self, sample):
        """Return model losses based on the input.

        :param pos_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param neg_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        triples = sample[:, :3].long()
        targets = sample[:, -1].double()
        scores = self(triples).mean(axis=1).double()

    def training_step(self, sample, batch_idx):
        scores, targets = sample
        loss = self.bce_loss(scores, targets)
        return {"loss": loss.mean()}

    def validation_step(self, sample, batch_idx):
        scores, targets = sample
        loss = self.bce_loss(scores, targets)
        return {"val_loss": loss.mean()}

    def train(self, x_train, y_train):
        """Apprentissage du modèle de ranking d'entités"""
        checkpoint_callback = ModelCheckpoint(
            filepath=str(self.model_path),
            verbose=True,
            monitor="checkpoint_on",
            mode="min",
        )
        self.epochs = 200

        self.batch_size = 256
        input_dim = x_train.shape[1]
        self.init_net(input_dim, hidden_size=512)

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=4,
            checkpoint_callback=checkpoint_callback,
        )
        trainer.fit(self, ds)

    def predict(self, m):
        return self.net(m)

    def load_model(self):
        """Chargement du modèle de ranking d'entités"""
        logger.info(f"EntityLinker loaded from {self.model_path}")
        self.load_from_checkpoint(self.model_path)
        logger.info(f"EntityLinker loaded")

    def save_model(self):
        """Sauvegarde du modèle de ranking d'entités"""
        self.save_checkpoint(self.model_path)

    def transform(self, X):
        """Extrait et pondère des entité d'un pour un ensemble de documents"""
        return [self.get_entities_dataframe(doc, v_doc) for doc, v_doc in X]

    def eval(self):
        """Evaluation du modèle de ranking d'entités"""
        from database.CorpusTransformer import generate

        annotation = None
        last_doc_id = None
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        i = 0
        formes = []
        for doc_id, text, entity_id, forme, start, end in generate(
            "annotation_eval_corpus"
        ):
            annotations = []
            if doc_id != last_doc_id:
                annotations = self.get_entities_dataframe(text, None)
                formes = []
            # calculer le silence qd une annotation prévu n'a pas été detectée (faux positif)
            for ann in annotations:
                i += 1
                if ann["entiteForme"] == forme and entity_id not in formes:
                    formes.append(entity_id)
                    success = entity_id == ann[self.id_label]
                    if not success:
                        fp += 1
                        logger.info(doc_id, forme, entity_id, ann[self.id_label])
                    else:
                        tp += 1
                    if (i % 500) == 0:
                        accuracy = tp / (tp + fp)
                        # recall =
                        logger.info(
                            f"positifs:{tp}, negatifs : {fp}, accuracy:{accuracy}"
                        )
                        # logger.info(f"precision:{precision} recall:{recall}")
            last_doc_id = doc_id

    def load_eval(self):
        self.load_model()
        self.eval()


if __name__ == "__main__":
    import fire

    fire.Fire(EntityLinker)


"""
TODO :
p_emb        : poincaré embeddings de la hierarchie de catégories Wikipédia + Annuaire
emb_cat(en)  : mean([p_emb(cat) for cat in categories(en)])
emb_rels(en) : mean([emb_en(en_r) for en_r in relations(en)])
emb_ctx(en)  : mean([word_emb(w) for w in contexte(en)])

emb(en) = concat(emb_cat, emb_rels, emb_ctx)

on peut ensuite faire une similarité entre chaque instance
puis sommer les embeddings identifiés en entrée d'un classifieur binaire
"""
