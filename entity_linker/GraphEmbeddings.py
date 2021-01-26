#%%
import numpy as np
from sklearn.utils import validation
import torch
from torch import tensor
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)
# https://github.com/jimmywangheng/knowledge_representation_pytorch/blob/master/model.py

# TODO :
# - utiliser un mlp pour encoder le doc vector emb_d
# - faire en sorte que emb_d=[1...1] si il n y a pas de document
# - sauvegarder l'embedding comme le produit (ent_embedding * doc_embedding)
# - integrer l'alignement d'embeddings dans l algo

# - directement exploiter l embedding du texte si il existe
# - sinon initiliser l'embedding à [1...1] ou random


#%%
class GraphEmbeddings(pl.LightningModule):
    def __init__(self, nb_ents, nb_rels, dim=300):
        super().__init__()
        self.save_hyperparameters()
        self.dim = dim
        self.norm = 2
        self.entity_count = nb_ents
        self.relation_count = nb_rels
        self.batch_size = 512

        # self.doc_emb =
        self.ent_emb = self._init_embs(nb_ents)
        self.rel_emb = self._init_embs(nb_rels)
        self.doc_emb = None  # self._init_embs(55563)
        self.drop = nn.Dropout(0.2)

        input_dim, hidden_size = 300, 300
        self.net = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            # nn.Dropout(p=0.15),
            # nn.Linear(input_dim, hidden_size),
            # nn.ReLU(),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=0.3),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )

        self.bce_loss = nn.BCELoss()


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-1)

    def _init_embs(self, nb_rows):
        emb = nn.Embedding(num_embeddings=nb_rows, embedding_dim=self.dim)
        # , scale_grad_by_freq=True
        # nn.init.xavier_uniform_(emb.weight.data)
        # emb.weight.data = F.normalize(emb.weight.data, p=self.norm, dim=1)
        return emb

    def load_pretrained(self, doc_embs):
        # self.doc_emb = emb_data
        self.doc_emb = torch.nn.Embedding.from_pretrained(doc_embs, freeze=True)
        # self.doc_emb.weight.requires_grad = False
        self.nb_docs, self.dim = doc_embs.shape
        print(self.nb_docs, self.dim)
        # print(self.doc_embeddings[[12171, 12172]])
        # self.doc_embeddings.weight.data = expr_emb_data
        # self.doc_embeddings.weight.requires_grad = False

    def embed(self, batch):
        # Which tokens in batch do not have representation, should have indices BIGGER
        # than the pretrained ones, adjust your data creating function accordingly
        mask = batch >= self.nb_docs
        pretrained_batch = batch.detach().clone()
        pretrained_batch[mask] = 0

        # Every token without representation has to be brought into appropriate range
        batch = batch - self.nb_docs
        # Zero out the ones which already have pretrained embedding
        batch[~mask] = 0
        embedded_batch = self.net(self.doc_emb(pretrained_batch))
        embedded_batch[mask] = self.ent_emb(batch)[mask]
        return embedded_batch
        # return self.net(embedded_batch)

    def forward(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets
        :param triplets: should have shape (B_size, 3) where dim 3 are head id, relation id, tail id.
        :return: dissimilarity score for given triplets
        """
        heads, rels, tails = tuple(triplets.T)
        return torch.sigmoid(self.embed(heads) + self.rel_emb(rels) - self.embed(tails))
        # return torch.sigmoid(self.embed(heads) * self.rel_emb(rels) * self.embed(tails))

    def forward_old(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets
        :param triplets: should have shape (B_size, 3) where dim 3 are head id, relation id, tail id.
        :return: dissimilarity score for given triplets
        """
        heads, rels, tails = tuple(triplets.T)
        return torch.sigmoid(
            self.drop(self.ent_emb(heads))
            * self.drop(self.doc_emb[heads])
            * self.rel_emb(rels)
            * self.drop(self.ent_emb(tails))
            * self.drop(self.doc_emb[tails])
        )

    def calculate_loss(self, sample):
        """Return model losses based on the input.

        :param pos_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param neg_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        triples = sample[:, :3].long()
        targets = sample[:, -1].double()
        scores = self(triples).mean(axis=1).double()
        return self.bce_loss(scores, targets)

    def training_step(self, sample, batch_idx):
        loss = self.calculate_loss(sample)
        return {"loss": loss.mean()}

    def validation_step(self, sample, batch_idx):
        loss = self.calculate_loss(sample)
        return {"val_loss": loss.mean()}


def train():
    from .BDCRelationsDataModule import BDCRelationsDataModule, IPTCDataModule

    # default used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath="/home/tgirault/data/reco/lightning_logs/",
        # save_top_k=True,
        verbose=True,
        monitor="checkpoint_on",
        mode="min",
        prefix="knotext",
    )

    ds = IPTCDataModule()  # BDCRelationsDataModule
    ds.prepare_data()
    embs = ds.get_doc_embeddings()
    model = GraphEmbeddings(ds.nb_cats, ds.nb_rels, dim=300)  # embs.shape[1])
    model.load_pretrained(embs)
    trainer = pl.Trainer(
        max_epochs=20,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=4,
        # checkpoint_callback=checkpoint_callback,
    )
    # auto_scale_batch_size=True,
    # shuffle=True,
    trainer.fit(model, ds)


#%%

if __name__ == "__main__":
    # train()
    import fire

    fire.Fire()
