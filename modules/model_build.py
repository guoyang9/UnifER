import torch
import torch.nn as nn
from transformers import ViltForQuestionAnswering, LxmertForQuestionAnswering, \
    VisualBertForQuestionAnswering, BertTokenizerFast
from modules.modal_embed import KnowledgeEmbedder


class KnowledgeNotNoise(nn.Module):
    def __init__(self, config, loss_fn, num_answers) -> None:
        """
        Args:
            config (str): default config file
            loss_fn (nn.Module): loss function for vqa, CE or BCE
            num_answers (int): number of candidate answers
        """
        super().__init__()
        self.config = config
        self.loss_fn = loss_fn
        self.num_answers = num_answers

        model = eval(config['transformer']['vlp_model']
            ).from_pretrained(config['transformer']['checkpoint_model'])
        self.vl_model = VLModel(model, name=config['transformer']['model_name'])

        self.sim_func = nn.CosineSimilarity(dim=-1)
        self.retriever = Retriever(config['train']['k_max_num'], self.sim_func)
        self.reader = Reader(config, self.vl_model, num_answers)

        # for retriever supervision
        self.retriever_loss = nn.MSELoss()
        self.reader_loss = loss_fn

    @torch.no_grad()
    def labeling(self, vlk_loss, vl_loss):
        """ Make labels for both reader and retriever. """
        diff = vl_loss.unsqueeze(dim=-1) - vlk_loss
        label_retriever = torch.tanh(diff)
        label_reader = torch.sigmoid(diff)
        return label_reader, label_retriever

    def forward(self, img, 
            question, inputs_question, 
            answer,
            knowledge_full, knowledge_embed):
        """
        Args:
            img (dict): img features
            question (str): raw question words
            inputs_question (dict): question related feature
            answer (tensor): ground-truth answer vectors
            knowledge_full (list): list of knowledge sentences
            knowledge_embed (tensor): same order with knowledge_full
        """
        vl = self.vl_model(img, inputs_question) # (b, hidden_size)

        topk_knowledge, topk_embed = self.retriever(vl, knowledge_embed, knowledge_full)
        vlk_logits, vl_logits = self.reader(img, question, topk_knowledge, vl)

        batch_size, topk = topk_embed.size()[:2]
        vl_loss = self.reader_loss(vl_logits, answer) # (b, )
        vlk_logits = vlk_logits.view(batch_size, topk, -1)
        vlk_loss = self.reader_loss(vlk_logits, answer.unsqueeze(dim=1))

        # disable instances that knowledge is ineffective
        effecting, match_label = self.labeling(vlk_loss, vl_loss)
        reader_loss = (vlk_loss * effecting)

        # simply using the match loss for retriever
        match_pred = self.sim_func(vl.unsqueeze(dim=1), topk_embed)
        retriever_loss = self.retriever_loss(match_pred, match_label)

        return retriever_loss.mean(), reader_loss.mean(), vlk_logits[:, 0, :]


class Retriever(nn.Module):
    def __init__(self, topk, sim_func) -> None:
        """
        Args:
            topk (int): return topk most relevant knowledge
            sim_func (nn.Module): similarity function, default cosine similarity
        """
        super().__init__()
        self.topk = topk 
        self.sim_func = sim_func

    @torch.no_grad()
    def forward(self, query, knowledge_embed, knowledge_full):
        """ Retrieve topk knoweldge in non-batch manner for resource reason. """ 
        topk_knowledge, topk_embed = [], []
        for query_single in query:
            sims = self.sim_func(query_single.unsqueeze(dim=0), knowledge_embed)
            indices = sims.topk(self.topk, dim=-1)[1]
            topk_embed.append(knowledge_embed[indices])
            topk_knowledge.append([knowledge_full[idx] for idx in indices])
        return topk_knowledge, torch.stack(topk_embed, dim=0)


class Reader(nn.Module):
    def __init__(self, config, vl_model, num_answers) -> None:
        super().__init__()
        self.vl_model = vl_model
        self.knowledge_embedder = KnowledgeEmbedder(
            eval(config['transformer']['tokenizer']),
            config['transformer']['checkpoint_token'], 
            max_q_len=config['train']['q_max_len'],
            max_k_len=config['train']['k_max_len'],
            max_k_num=config['train']['k_max_num'],
        )

        hidden_size = config['train']['hidden_size']
        fusion_modules = [nn.Dropout(config['run']['dropout'])]
        fusion_modules += list(vl_model.fusion_mlp.children())
        fusion_modules.append(nn.Dropout(config['run']['dropout']))
        self.fusion_mlp = nn.Sequential(*fusion_modules)

        if self.vl_model.name == 'visualbert':
            fusion_size = hidden_size
        else:
            fusion_size = hidden_size * 2
        self.classifier = nn.Parameter(torch.Tensor(fusion_size, num_answers))
        self.classifier.data.uniform_(-1 / hidden_size , 1 / hidden_size)

        self.result = []

    def l2_norm(self, input, dim=-1):
        norm = torch.norm(input, dim=dim, keepdim=True)
        output = torch.div(input, norm)
        return output

    def predict(self, feature, scale=16):
        """ mapped into cosine space
        Args:
            feature (tensor): final fused feature
            scale (int, optional): scale up logits (essential). Defaults to 16.
        """
        fusion = self.fusion_mlp(feature)
        fusion = self.l2_norm(fusion, dim=-1)
        kernel = self.l2_norm(self.classifier, dim=0)
        logits = scale * torch.mm(fusion, kernel)
        return logits

    def forward(self, img, question, knowledge, vl):
        """
        Args:
            img (PIL image): image pixels
            question (list): list of question words, (b, )
            knowledge (list): list of most relevant knowledge, (b, k, )
            vl (tensor): fused image-question feature, (b, h)
        """
        topk = len(knowledge[0])
        batch_size, img_size = img['feat'].size()[0], img['feat'].size()[1:]

        # conventional vqa model
        vl_logits = self.predict(vl)

        # incorporating knowledge info
        inputs_text = self.knowledge_embedder(knowledge, question) # (b * k, )
        img['feat'] = img['feat'].unsqueeze(dim=1).repeat(tuple([1, topk] + [1] * len(img_size))
            ).view(tuple([batch_size * topk] + list(img_size)))

        # enlarging image bbox features - for lxmert only
        if img['pos'] is not None:
            box_size = img['pos'].size()[1:]
            img['pos'] = img['pos'].unsqueeze(dim=1).repeat(tuple([1, topk] + [1] * len(box_size))
                ).view(tuple([batch_size * topk] + list(box_size)))

        vl_k = self.vl_model(img, inputs_text)
        vlk_logits = self.predict(vl_k)    
        return vlk_logits, vl_logits


class VLModel(nn.Module):
    def __init__(self, model, name) -> None:
        """ Build a general VLModel for knowledge-based VQA. 
        Args:
            name: currently support three ('vilt', 'lxmert', 'visualbert')    
        """
        super().__init__()
        self.model = model
        self.name = name
        if name == 'vilt':
            self.fusion_mlp = nn.Sequential(*list(model.classifier.children())[:-1])
        if name == 'lxmert':
            self.fusion_mlp = nn.Sequential(*list(model.answer_head.logit_fc.children())[:-1])
        if name == 'visualbert':
            self.fusion_mlp = model.dropout # only one dropout layer

    def forward(self, img, text):
        if self.name == 'vilt':
            inputs = {'pixel_values': img['feat']}
            inputs.update(text)
            outputs = self.model.vilt(**inputs)
            return outputs.pooler_output

        if self.name == 'lxmert':
            inputs = {
                'visual_feats': img['feat'],
                'visual_pos': img['pos'],
                'return_dict': True,
                'output_attentions': True
            }
            inputs.update(text)
            # lxmert_output = self.model.lxmert(**inputs)
            # return lxmert_output[2], lxmert_output.cross_encoder_attentions[-1]
            return self.model.lxmert(**inputs)[2]

        if self.name == 'visualbert':
            inputs = {'visual_embeds': img['feat']}
            inputs.update(text)
            return self.model.visual_bert(**inputs)[1]
