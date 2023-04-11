import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # prompts: # [n_cls, n_tkn, d_model] [160, 77, 512] tokenized_prompts: # (n_cls, n_tkn) (160, 77)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] shape is [batch, transformer.width(512)]
        # self.text_projection shape is [transformer.width, embed_dim] [512, 512]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # print(classnames)
        # print([_tokenizer.encode(name) for name in classnames])
        
        print("zhuoyan=== prompts len ", len(prompts)) # 351 or 160
        print(prompts[0])   # a photo of a cliff dwelling.
        print("name_lens[0:2] : ", name_lens[0], name_lens[1], name_lens[2]) # [2,1,2]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        print("zhuoyan=== tokenized_prompts ", tokenized_prompts.shape) #[351, 77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls # [500/351/160]
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, label):
        '''
        labels: shape (5,) used for indices what classes used for classification
        '''
        # prefix = self.token_prefix
        # suffix = self.token_suffix
        prefix = self.token_prefix[label.cpu()] # simple hack, only select 5 class embedding
        suffix = self.token_suffix[label.cpu()] # simple hack,

        ctx = self.ctx                     # (n_ctx, ctx_dim) [4, 512]
        # print("ctx dim: ", ctx.shape)
        bias = self.meta_net(im_features)  #  (batch, ctx_dim) [100, 512] or [1, 512]
        # bias = torch.zeros(5,512).half().cuda()
        # print("bias dim: ", bias.shape)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)  [100, 4, 512]
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            # ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)  #(n_cls, n_ctx, ctx_dim) ([351, 4, 512])
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(label.shape[0], -1, -1) # simple hack here #(n_cls_selected, n_ctx, ctx_dim) ([5, 4, 512])
            # print("ctx_i.shape", ctx_i.shape)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls_selected, n_tkn, ctx_dim) [5, 77, 512]
            # print("pts_i.shape", pts_i.shape)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)   # (batch_size, n_cls_selected, n_tkn, ctx_dim) [100, 5, 77, 512]
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None, exd_label = None):
        '''
        tensor([100, 194, 46, 277, 286], device='cuda:0') only encode these classes
        prompts: # [n_cls, n_tkn, d_model] [160, 77, 512]; tokenized_prompts: # (n_cls, n_tkn) (160, 77)
        consider n_cls_selected = 5, select 5 classes in one time
        '''
        # tokenized_prompts = self.tokenized_prompts
        print("label: ", label)
        print("exd_label: ", exd_label)
        if label is not None:
            label = label.cpu()
        tokenized_prompts = self.tokenized_prompts[label] # simple hack, shape (n_cls_selected, n_tkn) (5, 77)
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features, label)
        # print("zhuoyan===  prompt_learner: ", len(prompts))  # [batch_size]
        # print("prompts.shape", prompts.shape) # (batch_size, n_cls_selected, n_tkn, ctx_dim) [100, 5, 77, 512] [1,5,77,512]
        # print("image_features.shape", image_features.shape) # (batch, ctx_dim) [100,512] [1,512]
        logits = []
        for pts_i, imf_i in zip(prompts, image_features): # num_batch
            # print("pts_i.shape: {}, imf_i.shape: {}".format(pts_i.shape, imf_i.shape))
            text_features = self.text_encoder(pts_i, tokenized_prompts) # [n_cls, ctx_dim] [5, 512]
            # print("text_features.shape: ", text_features.shape)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t() # text_feature.T: [512, n_cls], imf_i: [512,]
            logits.append(l_i)
        logits = torch.stack(logits) ###
        print("logits.shape: ", logits.shape)
        if self.prompt_learner.training:
            # return F.cross_entropy(logits, label)
            return F.cross_entropy(logits, exd_label.cuda())
            # return F.cross_entropy(logits, torch.arange(label.shape[0]).cuda()) # do not simple hack, only classify among 5 classes
        
        return logits


@TRAINER_REGISTRY.register()
class MTCoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames ###
        print("zhuoyan=== : classnames.shape", len(classnames))

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        print("zhuoyan == forward_backward label: ", label)  # tensor([100, 194,  46, 277, 286], device='cuda:0')
        # model will need to recognize 15 labels, and only classify among this classes!
        exd_label =  torch.arange(label.shape[0])

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label, exd_label) # exd_label is the true label used for CrossEntropy
            optim.zero_grad()
            if loss.shape: # loss is a vector in distributed training
                loss = loss.mean()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        # print("zhuoyan == parse_batch_train: ", label)
        # label =  torch.arange(label.shape[0])
        # print("zhuoyan == parse_batch_train: ", label)
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
