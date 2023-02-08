import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import random
import pickle
import argparse
from fairseq.models.transformer import (
    TransformerEncoder,
    Linear
)
import math
from typing import Dict, List, Optional
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import TransformerConfig
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange
from functools import partial
from itertools import islice, cycle
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from g_mlp_pytorch import gMLPBlock
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from models import (
    FairseqEncoder,
    FairseqDecoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerDecoder,
    Embedding,
)
from fairseq.models.fairseq_model import check_type

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


def cast_tuple(val, depth=1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth


class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True)
        return x / maxes


class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            seq_len,
            reversible=False,
            causal=True,
            heads=8,
            dim_head=64,
            ff_mult=4,
            attn_dropout=0.,
            ff_dropout=0.,
            attn_types=None,
            image_fmap_size=None,
            sparse_attn=False,
            stable=False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for ind, sparse_attn, attn_type in zip(range(depth), sparse_layer, attn_type_layer):
            if attn_type == 'full':
                attn_class = partial(Attention, stable=stable)
            elif attn_type == 'sparse':
                attn_class = SparseAttention
            elif attn_type == 'axial_row':
                attn_class = partial(SparseAxialCausalAttention, seq_len=seq_len, axis=0, image_size=image_fmap_size,
                                     stable=stable)
            elif attn_type == 'axial_col':
                attn_class = partial(SparseAxialCausalAttention, seq_len=seq_len, axis=1, image_size=image_fmap_size,
                                     stable=stable)
            elif attn_type == 'conv_like':
                attn_class = partial(SparseConvCausalAttention, seq_len=seq_len, image_size=image_fmap_size,
                                     stable=stable)
            elif attn_type == 'mlp':
                attn_class = partial(gMLPBlock, seq_len=seq_len)
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')

            if attn_type != 'mlp':
                attn = attn_class(dim, causal=causal, seq_len=seq_len, heads=heads, dim_head=dim_head,
                                  dropout=attn_dropout)
            else:
                attn = attn_class(dim=dim, causal=causal, dim_ff=dim * 4)

            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn)),
                LayerScale(dim, ind + 1, PreNorm(dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout)))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}

        self.layers = execute_type(layers, args_route=attn_route_map)

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    def inner(*args, **kwargs):
        return val

    return inner


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def build_node_vocab(args):
    stopwords_dir = args.stopwords_dir
    src_dict_dir = args.src_dict_dir
    src_en_dir = args.src_en_dir
    tfidf = args.tfidf
    num_img = args.num_img
    cap2image_file = args.cap2image_file
    total_img = 0
    stop_words = []
    if stopwords_dir:
        with open((stopwords_dir), "r") as data:
            for word in data:
                stop_words.append(word.strip())
    src_dict = {}
    with open((src_dict_dir), "r") as data:
        for line in data:
            word, idx = line.strip().split()
            src_dict[word] = int(idx)
    obj2ids = {}
    cap_sentences = []
    cap_sentences_raw = []
    with open((src_en_dir), "r") as data:
        for line in data:
            cap = line.strip()
            total_img += 1
            if stopwords_dir:
                wordsFiltered = []
                cap = cap.strip().split()
                for w in cap:
                    if w not in stop_words:
                        wordsFiltered.append(w)
                cap = " ".join(wordsFiltered)
            cap_sentences.append(cap.split())
            cap_sentences_raw.append(cap)

    n = tfidf
    words, weight = None, None
    if n > 0:
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(cap_sentences_raw))
        words = vectorizer.get_feature_names()
        weight = tfidf.toarray()

    for idx, cap in enumerate(cap_sentences):
        if n > 0:
            w = weight[idx]
            loc = np.argsort(-w)
            top_words = []
            for i in range(n):
                if w[loc[i]] > 0.0:
                    top_words.append(words[loc[i]])
            top_cap = []
            cap = cap
            for word in cap:
                if word.lower() in top_words:
                    top_cap.append(word)
            cap = top_cap

        tokenized_cap = cap

        for cap in tokenized_cap:
            if cap not in stop_words and cap in src_dict:
                if src_dict[cap] not in obj2ids:
                    obj2ids[src_dict[cap]] = [idx + 1]  # index 0 is used for placeholder
                else:
                    cap_list = obj2ids[src_dict[cap]]
                    cap_list.append(idx + 1)
                    obj2ids[src_dict[cap]] = cap_list

    for key, value in obj2ids.items():
        if len(value) < num_img:
            value.extend([0] * (num_img - len(value)))
            obj2ids[key] = value
        else:
            value = random.sample(value, num_img)
            obj2ids[key] = value

    pickle.dump(obj2ids, open(cap2image_file, "wb"))

    # print("data process finished!")
    # print(len(obj2ids))
    # print(total_img)


class VSGHallucinator(nn.Module):
    def __init__(
            self,
            arg,
            dim,
            num_text_tokens=10000,
            text_seq_len=256,
            heads=8,
            dim_head=64,
            reversible=False,
            attn_dropout=0.,
            ff_dropout=0,
            sparse_attn=False,
            attn_types=None,
            loss_img_weight=7,
            stable=False,
            bos_idx=0,
            pad_idx=0
    ):
        super().__init__()

        num_image_tokens = arg.num_tokens
        image_fmap_size = (arg.image_size // (2 ** arg.num_layers))
        image_seq_len = image_fmap_size ** 2

        num_text_tokens = num_text_tokens + text_seq_len
        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim)  # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(image_fmap_size, image_fmap_size))

        self.num_text_tokens = num_text_tokens  # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.transformer = Transformer(
            dim=dim,
            causal=True,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            reversible=reversible,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            image_fmap_size=image_fmap_size,
            sparse_attn=sparse_attn,
            stable=stable
        )

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        logits_mask = (
                ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
                ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )

        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight

        self.bos_idx = bos_idx
        self.pad_idx = pad_idx

    @torch.no_grad()
    @eval_decorator
    def generate_images(
            self,
            text,
            *,
            clip=None,
            mask=None,
            filter_thres=0.5,
            temperature=1.,
            img=None,
            num_init_img_tokens=None,
            ret_seq=False
    ):
        vis, text_seq_len, image_seq_len, num_text_tokens = self.vis, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        if text.shape[-1] < self.text_seq_len:
            pad_len = self.text_seq_len - text.shape[-1]
            text = F.pad(text, (0, pad_len), value=self.pad_idx)

        text = text[:, :text_seq_len]
        out = text

        if exists(img):
            image_size = vis.image_size
            assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[
                3] == image_size, f'input image must have the correct image size {image_size}'

            indices = vis.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens,
                                     int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'

            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim=-1)

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image, mask=mask)[0][:, :, -1]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            sample -= (
                num_text_tokens if is_image else 0)  # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

        text_seq = out[:, :text_seq_len]

        hal_vis_rep = out[:, -image_seq_len:]
        if ret_seq:
            return hal_vis_rep

        images = vis.decode(hal_vis_rep)

        if exists(clip):
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images

    def forward(
            self,
            text,
            image=None,
            mask=None,
            return_loss=False
    ):
        # pad input to target length
        if text.shape[-1] < self.text_seq_len:
            pad_len = self.text_seq_len - text.shape[-1]
            text = F.pad(text, (0, pad_len), value=self.pad_idx)
        if text.shape[-1] > self.text_seq_len:
            text = text[..., :self.text_seq_len]
        device, total_seq_len = text.device, self.total_seq_len
        text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == self.pad_idx, text_range, text)
        text = F.pad(text, (1, 0), value=self.bos_idx)
        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))
        seq_len = tokens.shape[1]
        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4
            if is_raw_image:
                image_size = self.vis.image_size
                assert tuple(image.shape[1:]) == (
                    3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'
                image = self.vis.get_codebook_indices(image)
            image_len = image.shape[1]
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(image_emb)
            tokens = torch.cat((tokens, image_emb), dim=1)
            seq_len += image_len
        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]
        out = self.transformer(tokens)
        if self.stable:
            out = self.norm_by_max(out)
        logits = self.to_logits(out)
        logits_mask = self.logits_mask[:, :seq_len]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)
        logits = rearrange(logits, 'b n c -> b c n')
        logits_text = logits[:, :, :self.text_seq_len]
        vis_rep = logits[:, :, self.text_seq_len:]
        if not return_loss:
            return vis_rep, None

        assert exists(image), 'when training, image must be supplied'
        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim=1)
        labels_text, labels_img = labels[:, :self.text_seq_len], labels[:, self.text_seq_len:]
        loss_text = F.cross_entropy(logits_text, labels_text)
        loss_img = F.cross_entropy(vis_rep, labels_img)
        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
        return vis_rep, loss


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerJointEncoderBase':
        return 'TransformerJointEncoder'
    else:
        return module_name


class PositionalEmbedding2D(nn.Module):
    '''
    2D positional embedding for image grid features.
    num_rows: Number of rows
    num_cols: Number of columns (=num_rows by default)
    embedding_dim: Embedding dimension
    learned: Use learned positional embeddings instead of sinusoidals
    use_2d: Embed row and column indices separately
        - 'sum': add row and column embeddings, each with dim = embedding_dim
        - 'concat': concatenate row and column embeddings, each with dim = embedding_dim / 2
        - 'none': use 1d positional embedding with flattened features
    '''

    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 embedding_dim: int,
                 padding_idx: int,
                 learned: bool = False,
                 use_2d: float = 'sum'
                 ):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.seq_len = self.num_rows * self.num_cols
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.learned = learned
        self.use_2d = use_2d
        self._build_pos_encoder()

    def _build_pos_encoder(self):
        if self.use_2d in ['sum', 'concat']:
            if self.use_2d == 'concat':
                assert self.embedding_dim % 2 == 0, 'Embedding dim must be even for concatenated embedding'
                pos_embed_dim = self.embedding_dim // 2
            else:
                pos_embed_dim = self.embedding_dim
            self.embed_row = PositionalEmbedding(
                self.num_rows,
                pos_embed_dim,
                self.padding_idx,
                learned=self.learned,
            )
            self.embed_col = PositionalEmbedding(
                self.num_cols,
                pos_embed_dim,
                self.padding_idx,
                learned=self.learned,
            )
        else:
            self.embed = PositionalEmbedding(
                self.seq_len,
                self.embedding_dim,
                self.padding_idx,
                learned=self.learned,
            )

    def forward(
            self,
            input,
            **kwargs
    ):
        # input shape: bs * (h * w) * embed_dim
        assert input.shape[1] == self.seq_len, 'Input shape mismatch'
        assert input.shape[2] == self.embedding_dim, 'Input dimension mismatch'
        bs, seq_len, embed_dim = input.shape
        if self.use_2d in ['sum', 'concat']:
            if self.use_2d == 'concat':
                embed_dim = embed_dim // 2
            dummy_input_row = torch.ones(1, self.num_rows).to(input.device)
            dummy_input_col = torch.ones(1, self.num_cols).to(input.device)
            pos_embed_row = self.embed_row(dummy_input_row, **kwargs)  # 1 x num_rows x embed_dim
            pos_embed_col = self.embed_col(dummy_input_col, **kwargs)  # 1 x num_cols x embed_dim
            pos_embed_row = pos_embed_row.unsqueeze(2).expand(1, self.num_rows, self.num_cols,
                                                              embed_dim)  # 1 x num_rows x num_cols x embed_dim
            pos_embed_col = pos_embed_col.unsqueeze(1).expand(1, self.num_rows, self.num_cols,
                                                              embed_dim)  # 1 x num_rows x num_cols x embed_dim
            if self.use_2d == 'sum':
                pos_embed = pos_embed_row + pos_embed_col
            else:
                pos_embed = torch.cat([pos_embed_row, pos_embed_col], -1)
            pos_embed = pos_embed.flatten(1, 2)
        else:
            dummy_input = torch.ones(1, seq_len).to(input.device)
            pos_embed = self.embed(dummy_input, **kwargs)
        return pos_embed


class MMJointEncoderBase(FairseqEncoder):

    def __init__(self, cfg, args, dictionary, embed_tokens, vis_embed_tokens):
        self.cfg = cfg
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.vis_shape = (args.vis_encoder_grid_h, args.vis_encoder_grid_w)
        self.vis_pool = nn.AdaptiveAvgPool2d(self.vis_shape)
        self.vis_positions = args.vis_encoder_grid_h * args.vis_encoder_grid_w
        self.max_source_positions = cfg.max_source_positions + self.vis_positions

        self.embed_tokens = embed_tokens
        self.vis_embed_tokens = vis_embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        self.vis_embed_positions = (
            PositionalEmbedding2D(
                args.vis_encoder_grid_h,
                args.vis_encoder_grid_w,
                embedding_dim=args.vislang_embed_dim,
                padding_idx=0,
                learned=args.vis_encoder_learned_pos,
                use_2d=args.vis_encoder_use_2d_pos
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
            self,
            src_tokens,
            vis_input,
            token_embedding: Optional[torch.Tensor] = None
    ):
        # embed visual tokens
        def vis_embed(emb_v):
            if self.vis_embed_tokens is not None:
                if len(emb_v.shape) == 2:
                    vis_token_embedding = self.vis_embed_tokens(emb_v)
                else:
                    vis_token_embedding = emb_v @ self.vis_embed_tokens.weight
                emb_v = self.embed_scale * vis_token_embedding
            return emb_v

        # text embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)

        # visual embed tokens
        if isinstance(vis_input, list):
            emb_v = [vis_embed(x) for x in vis_input]
            emb_v = torch.stack(emb_v, dim=0).mean(0)
        else:
            emb_v = vis_embed(vis_input)

        # visual positional embeddings
        if len(emb_v.shape) == 4:
            # resize feature maps to self.vis_shape
            if emb_v.shape[1:3] != self.vis_shape:
                emb_v = emb_v.permute(0, 3, 1, 2)  # NHWC -> NCHW
                emb_v = self.vis_pool(emb_v)
                emb_v = emb_v.permute(0, 2, 3, 1)  # NCHW -> NHWC
            # flatten transformer input
            emb_v = emb_v.flatten(1, 2)  # N(HW)C
        if self.vis_embed_positions is not None:
            emb_v = emb_v + self.vis_embed_positions(emb_v)

        # vision-language concatenation
        if hasattr(self.args, 'vis_only') and self.args.vis_only:
            x = emb_v
        else:
            x = torch.cat([x, emb_v], 1)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            vis_input,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, vis_input, src_lengths, return_all_hiddens, token_embeddings
        )

    def forward_scriptable(
            self,
            src_tokens,
            vis_input,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, vis_input, token_embeddings)
        bs, seqlen = x.shape[:2]
        vis_padding_mask = torch.zeros(bs, seqlen - encoder_embedding.shape[1]).to(encoder_padding_mask.device).bool()
        encoder_padding_mask = torch.cat([encoder_padding_mask, vis_padding_mask], 1)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None or self.vis_embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions,
                   self.embed_positions.max_positions + self.vis_embed_positions.seq_len)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


def unfold2d(x, kernel_size, stride, padding):
    ### using torch.nn.functional.unfold is also okay and the effiency will be compared later.

    x = F.pad(x, [padding] * 4)
    bs, in_c, h, w = x.size()
    ks = kernel_size
    strided_x = x.as_strided((bs, in_c, (h - ks) // stride + 1, (w - ks) // stride + 1, ks, ks),
                             (in_c * h * w, h * w, stride * w, stride, w, 1))
    return strided_x


class CosSim2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, eps=1e-12, bias=True):
        super(CosSim2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = padding

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_normal_(w)
        self.w = nn.Parameter(w.view(out_channels, in_channels, -1), requires_grad=True)

        self.p = nn.Parameter(torch.empty(out_channels))
        nn.init.constant_(self.p, 2)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None

    def sigplus(self, x):
        return nn.Sigmoid()(x) * nn.Softplus()(x)

    def forward(self, x):
        x = unfold2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)  # nchwkk
        n, c, h, w, _, _ = x.shape
        x = x.reshape(n, c, h, w, -1)
        x = F.normalize(x, p=2.0, dim=-1, eps=self.eps)

        w = F.normalize(self.w, p=2.0, dim=-1, eps=self.eps)
        x = torch.einsum('nchwl,vcl->nvhw', x, w)
        sign = torch.sign(x)

        x = torch.abs(x) + self.eps
        x = x.pow(self.sigplus(self.p).view(1, -1, 1, 1))
        # pdb.set_trace()
        x = sign * x

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


class FusionMappingEncoder(MMJointEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, vis_embed_tokens):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            args,
            dictionary,
            embed_tokens,
            vis_embed_tokens
        )
        self.tgt_dict = dictionary
        self.matcher = CosSim2d(args.in_dim, args.out_dim, args.kernel_size, args.stride, padding=3).cuda()
        self.f_encoder = GCNWraper(args, dictionary, embed_tokens)

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )

    def forward_scriptable(self, args, v_rep, l_rep):
        mat_rep = self.matcher(v_rep, l_rep)
        r_rep = self.f_encoder(args, self.tgt_dict, mat_rep)

        # x, encoder_embedding = self.forward_embedding(src_tokens, vis_input, token_embeddings)
        bs, seqlen = r_rep.shape[:2]
        # vis_padding_mask = torch.zeros(bs, seqlen - r_rep.shape[1]).to(encoder_padding_mask.device).bool()
        out_enc_rep = torch.cat([mat_rep, bs], 1)

        out_enc_rep = out_enc_rep.transpose(0, 1)

        if self.layer_norm is not None:
            out_enc_rep = self.layer_norm(out_enc_rep)

        return {
            "encoder_out": [out_enc_rep],
        }


class GraphConvNetworkLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphConvNetworkLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), adj
        else:
            return h_prime, adj

    @property
    def scaler(self):
        return self._scaler.exp()

    @property
    def rel_weight(self):
        if self.weight_act == 'sigmoid':
            return torch.sigmoid(self._rel_weight)
        elif self.weight_act == 'softmax':
            return torch.softmax(self._rel_weight, dim=-1)

    def init_weights(self):
        """Initialize token embedding and output bias."""
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'pos_emb'):
            self.pos_emb.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.fill_(0)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_features, out_features, layers, dropout, alpha):
        super(GCN, self).__init__()
        self.layers = layers

        self.GCN_layers = nn.ModuleList([
            GraphConvNetworkLayer(in_features, out_features, dropout, alpha)
            for _ in range(layers)])

    def forward(self, h, adj):
        e, adj = self.GCN_layers(h, adj)
        return e


class GCNWraper(nn.Module):
    """
    GCN encoder or decoder.
    """

    def __init__(self, args, src_dict, embed_tokens):
        self.args = args
        super().__init__()
        self.vislang_embed_norm = args.vislang_embed_norm
        self.mmt_inference = args.mmt_inference

    def normalize(self, feat, mode):
        if mode == 'none':
            pass
        elif mode == 'l1':
            feat = F.normalize(feat, p=1, dim=-1)
        elif mode == 'l2':
            feat = F.normalize(feat, p=2, dim=-1)
        else:
            raise Exception('normalization mode not supported:', mode)
        return feat

    def forward(self, vis_input, src_tokens, src_lengths):
        ret_v = vis_input is not None
        ret_l = (self.encoder_l is not None) and (self.training or not self.mmt_inference)
        assert ret_v or ret_l, 'Neither visual nor language features are available'

        # extract features from visual input
        if ret_v:
            emb_v = self.encoder_v(vis_input)
            if len(emb_v.shape) == 4:
                emb_v = emb_v.permute(0, 2, 3, 1)  # NCHW -> NHWC

            # if using pretrained codebook, project to target dimension and normalize
            if self.proj_v is not None:
                emb_v = self.proj_v(emb_v)
                emb_v = self.normalize(emb_v, self.vislang_embed_norm)
        else:
            emb_v = None

        # hallucinate visual features from text input
        if ret_l:
            if self.training:
                emb_l, loss = self.encoder_l(src_tokens, emb_v)  # single forward pass at training time
            else:
                emb_l, loss = self.encoder_l(src_tokens)  # autoregressive decoding at test time
            if self.proj_l is not None:
                if isinstance(emb_l, list):
                    emb_l = [self.proj_l(x) for x in emb_l]
                    emb_l = [self.normalize(x, self.vislang_embed_norm) for x in emb_l]
                else:
                    emb_l = self.proj_l(emb_l)
                    emb_l = self.normalize(emb_l, self.vislang_embed_norm)
        else:
            emb_l = loss = None

        return {
            "emb_v": emb_v,
            "emb_l": emb_l,
            "loss": loss
        }


class VGCNWraper(GCNWraper):
    def __init__(self, args, src_dict, embed_tokens):
        self.args = args
        super().__init__(args, src_dict, embed_tokens)
        self.encoder_v, self.proj_v = self.build_vis_encoder(args)
        # self.encoder_l, self.proj_l = self.build_lang_encoder(args, src_dict, embed_tokens, self.encoder_v)
        self.vislang_embed_norm = args.vislang_embed_norm
        self.mmt_inference = args.mmt_inference

    def build_vis_encoder(self, args):
        if args.vis_base_encoder == 'RCNN':
            self.base_model = vision.__dict__[args.vis_encoder_arch](
                model_path=args.vis_encoder_model_path,
                config_path=args.vis_encoder_config_path,
                codebook=args.vis_encoder_use_codebook,
            )
        self.gcn_model = GraphConvNetworkLayer(args.input_encoding_size, args.size, self.drop_prob_lm)
        if not args.vis_encoder_finetune:
            for param in self.gcn_model.parameters():
                param.requires_grad = False
        if args.vis_encoder_use_codebook:
            proj = Linear(args.vis_encoder_embed_dim, args.vislang_embed_dim, bias=False)
        else:
            proj = None
        return self.base_model, self.gcn_model, proj


class LGCNWraper(GCNWraper):
    def __init__(self, args, src_dict, embed_tokens):
        self.args = args
        super().__init__(args, src_dict, embed_tokens)
        self.encoder_l, self.proj_l = self.build_lang_encoder(args, src_dict, embed_tokens)
        self.vislang_embed_norm = args.vislang_embed_norm
        self.mmt_inference = args.mmt_inference

    def build_lang_encoder(self, args):
        if args.vis_base_encoder == 'Trm':
            self.base_model = TransformerEncoder(args, args.src_dict, args.embed_tokens)
            proj = Linear(args.encoder_embed_dim, args.vislang_embed_dim, bias=False)
        self.gcn_model = GraphConvNetworkLayer(args.input_encoding_size, args.size, self.drop_prob_lm)
        if not args.vis_encoder_finetune:
            for param in self.gcn_model.parameters():
                param.requires_grad = False
        if args.vis_encoder_use_codebook:
            proj = Linear(args.vis_encoder_embed_dim, args.vislang_embed_dim, bias=False)
        else:
            proj = None
        return self.base_model, self.gcn_model, proj


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features1, features2, labels=None, mask=None):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features1.is_cuda
                  else torch.device('cpu'))

        if len(features1.shape) < 3 or len(features2.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features1.shape) > 3 or len(features2.shape) > 3:
            features1 = features1.view(features1.shape[0], features1.shape[1], -1)
            features2 = features2.view(features2.shape[0], features2.shape[1], -1)

        batch_size = features1.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features1.shape[1]
        contrast_feature = torch.cat(torch.unbind(features1, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features1[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        contrast_count2 = features2.shape[1]
        contrast_feature2 = torch.cat(torch.unbind(features2, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature2 = features2[:, 0]
            anchor_count2 = 1
        elif self.contrast_mode == 'all':
            anchor_feature2 = contrast_feature2
            anchor_count2 = contrast_count2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_dot_contrast2 = torch.div(
            torch.matmul(anchor_feature2, contrast_feature2.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast + anchor_dot_contrast2, dim=1, keepdim=True)
        logits = anchor_dot_contrast + anchor_dot_contrast2 - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count + anchor_count2, contrast_count + contrast_count2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class BaseHallucUMMT(BaseFairseqModel):
    """Base class for hallucination multi-modal encoder-decoder models.

    Args:
        encoder_vl (FairseqEncoder): the joint vision-language encoder
        encoder_v (nn.Module): the vision encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder_vl, encoder_v, decoder):
        super().__init__()

        self.encoder_vl = encoder_vl
        self.encoder_v = encoder_v
        self.decoder = decoder

        check_type(self.encoder_vl, FairseqEncoder)
        check_type(self.encoder_v, nn.Module)
        check_type(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, vis_input=None, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            vis_input (FloatTensor): visual input, optional

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_v_out = self.encoder_v(vis_input, src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = {}

        # multimodal stream using extracted visual features
        halluc_only = hasattr(self.args, 'halluc_only') and self.args.halluc_only
        if (not halluc_only) and encoder_v_out['emb_v'] is not None:
            encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_v'], src_lengths=src_lengths, **kwargs)
            decoder_out['vislang'] = self.decoder(
                prev_output_tokens, encoder_out=encoder_vl_out, **kwargs
            )
        else:
            decoder_out['vislang'] = None

        # hallucination stream using text features only
        if encoder_v_out['emb_l'] is not None:
            if isinstance(encoder_v_out['emb_l'], list):
                encoder_vl_out = [self.encoder_vl(src_tokens, x, src_lengths=src_lengths, **kwargs) for x in
                                  encoder_v_out['emb_l']]
                decoder_out_list = [self.decoder(prev_output_tokens, encoder_out=x, **kwargs) for x in encoder_vl_out]
                decoder_out_list, decoder_out_extra = zip(*decoder_out_list)
                decoder_out_avg = torch.stack(decoder_out_list, dim=0).mean(0)
                decoder_out['halluc'] = (decoder_out_avg, decoder_out_extra)
            else:
                encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_l'], src_lengths=src_lengths, **kwargs)
                decoder_out['halluc'] = self.decoder(prev_output_tokens, encoder_out=encoder_vl_out, **kwargs)

            decoder_out['loss'] = None if halluc_only else encoder_v_out['loss']
        else:
            return decoder_out['vislang']

        return decoder_out

    def forward_encoder(self, src_tokens, src_lengths, prev_output_tokens, vis_input=None, **kwargs):
        encoder_v_out = self.encoder_v(vis_input, src_tokens, src_lengths=src_lengths, **kwargs)
        if isinstance(encoder_v_out, dict):
            # use hallucinated features at inference time
            if 'emb_l' in encoder_v_out and encoder_v_out['emb_l'] is not None:
                encoder_v_out = encoder_v_out['emb_l']
            else:
                encoder_v_out = encoder_v_out['emb_v']
        if hasattr(self.args, 'rand_inference') and self.args.rand_inference:
            encoder_v_out = torch.randint_like(encoder_v_out, self.args.vis_encoder_tokens)
        encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out, src_lengths=src_lengths, **kwargs)
        return encoder_vl_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, vis_input=None, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_v_out = self.encoder_v(vis_input, src_tokens, src_lengths=src_lengths, **kwargs)
        features = {}

        # multimodal stream using extracted visual features
        if encoder_v_out['emb_v'] is not None:
            encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_v'], src_lengths=src_lengths, **kwargs)
            features['vislang'] = self.decoder.extract_features(
                prev_output_tokens, encoder_out=encoder_vl_out, **kwargs
            )
        else:
            features['vislang'] = None

        # hallucination stream using text features only
        if encoder_v_out['emb_l'] is not None:
            encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_l'], src_lengths=src_lengths, **kwargs)
            features['halluc'] = self.decoder.extract_features(
                prev_output_tokens, encoder_out=encoder_vl_out, **kwargs
            )
            features['loss'] = encoder_v_out['loss']
        else:
            return features['vislang']

        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder_vl.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


@register_model("SGHallucUMMT")
class SGHallucUMMT(BaseHallucUMMT):

    def __init__(self, args, encoder_vl, encoder_v, decoder):
        super().__init__(encoder_vl, encoder_v, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-gcn-dim', type=int, metavar='N',
                            help='encoder gcn dimension')
        parser.add_argument('--encoder-gcn-layers', type=int, metavar='N',
                            help='encoder gcn layers')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for vision encoder
        parser.add_argument('--vis-encoder-arch', type=str,
                            help='vision encoder architecture')
        parser.add_argument('--vis-encoder-model-path', type=str,
                            help='path to vision encoder checkpoint')
        parser.add_argument('--vis-encoder-config-path', type=str,
                            help='path to vision encoder configs')
        parser.add_argument('--vis-encoder-finetune', action='store_true',
                            help='finetune visual encoder')
        parser.add_argument('--vis-encoder-use-codebook', action='store_true',
                            help='use pretrained visual codebook')
        parser.add_argument('--vis-encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained visual encoder embedding')
        parser.add_argument('--vis-encoder-embed-dim', type=int,
                            help='vision encoder feature dimension')
        parser.add_argument('--vis-encoder-tokens', type=int,
                            help='vision encoder vocabulary size')
        parser.add_argument('--vis-encoder-grid-h', type=int,
                            help='vision encoder feature grid height')
        parser.add_argument('--vis-encoder-grid-w', type=int,
                            help='vision encoder feature grid width')
        parser.add_argument('--vis-encoder-learned-pos', action='store_true',
                            help='use learned positional embedding for visual features')
        parser.add_argument('--vis-encoder-use-2d-pos', type=str,
                            help='use 2d positional embedding for visual features')
        parser.add_argument('--vis-encoder-hallucinate', type=str,
                            help='hallucinate vision encoder during training')
        parser.add_argument('--vis-only', action='store_true',
                            help='use visual input only, ignoring input sentence')
        # args for hallucination
        parser.add_argument('--halluc-model-path', type=str,
                            help='path to hallucination transformer checkpoint')
        parser.add_argument('--halluc-args', type=str,
                            help='hallucination transformer args')
        parser.add_argument('--mmt-inference', action='store_true',
                            help='use ground-truth image embeddings at test time')
        parser.add_argument('--rand-inference', action='store_true',
                            help='use random image embeddings at test time')
        parser.add_argument('--halluc-only', action='store_true',
                            help='use hallucination stream only')
        parser.add_argument('--pretrain-mmt', type=str,
                            help='path to load pretrained MMT transformer')
        # args for vision-language contrastive learning
        parser.add_argument('--vislang-embed-dim', type=int,
                            help='multimodal embedding dimension')
        parser.add_argument('--vislang-embed-norm', type=str,
                            help='normalize multimodal embedding (l1/l2/none)')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            lan_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = lan_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            lan_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if not args.vis_encoder_use_codebook:
            vis_embed_tokens = nn.Embedding(args.vis_encoder_tokens, args.encoder_embed_dim, padding_idx=None)
            nn.init.normal_(vis_embed_tokens.weight, mean=0, std=args.encoder_embed_dim ** -0.5)
        else:
            vis_embed_tokens = None
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        # encoder_vl = cls.build_lang_encoder(args, src_dict, lan_embed_tokens, vis_embed_tokens)
        encoder_v = cls.build_vis_encoder(args, src_dict, vis_embed_tokens)
        encoder_l = cls.build_lang_encoder(args, src_dict, lan_embed_tokens)
        hall_module = cls.build_vsg_halluc(args)
        fmencoder = cls.build_vislang_encoder(args, src_dict, vis_embed_tokens, lan_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        decoderTrm = cls.build_decoderTrm(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder_l = fsdp_wrap(encoder_v, min_num_params=min_params_to_wrap)
            encoder_v = fsdp_wrap(encoder_l, min_num_params=min_params_to_wrap)
            hall_module = fsdp_wrap(hall_module, min_num_params=min_params_to_wrap)
            fmencoder = fsdp_wrap(fmencoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
            decoderTrm = fsdp_wrap(decoderTrm, min_num_params=min_params_to_wrap)
        return cls(args, encoder_l, encoder_v, hall_module, fmencoder, decoder, decoderTrm)

    @classmethod
    def build_vsg_halluc(cls, args):
        halluc_args = json.loads(args.halluc_args)
        hall_module = VSGHallucinator(args, args.dim, halluc_args)
        return hall_module

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_vislang_encoder(cls, args, src_dict, v_rep, l_rep):
        fmencoder = FusionMappingEncoder(args, src_dict, v_rep, l_rep)
        return fmencoder

    @classmethod
    def build_lang_encoder(cls, args, src_dict, lan_embed_tokens):
        encoder = LGCNWraper(args, src_dict, lan_embed_tokens)
        if hasattr(args, 'pretrain_mmt') and args.pretrain_mmt is not None:
            pref = 'encoder_vl.'
            ckp = torch.load(args.pretrain_mmt)['model']
            enc_ckp = {k[len(pref):]: v for k, v in ckp.items() if k.startswith(pref)}
            encoder.load_state_dict(enc_ckp)
        return encoder

    @classmethod
    def build_vis_encoder(cls, args, src_dict, embed_tokens):
        encoder = VGCNWraper(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LGCNWraper(args, tgt_dict, embed_tokens)
        return decoder

    @classmethod
    def build_decoderTrm(cls, args, tgt_dict, embed_tokens):
        decoderTrm = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        if hasattr(args, 'pretrain_mmt') and args.pretrain_mmt is not None:
            pref = 'decoder.'
            ckp = torch.load(args.pretrain_mmt)['model']
            dec_ckp = {k[len(pref):]: v for k, v in ckp.items() if k.startswith(pref)}
            decoderTrm.load_state_dict(dec_ckp)
        return decoderTrm


@register_model_architecture("SGHallucUMMT", "SGHallucUMMT")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_gcn_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_gcn_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # vision encoder args
    args.vis_encoder_arch = getattr(args, "vis_encoder_arch", "vqgan")
    # args.vis_encoder_args = getattr(args, "vis_encoder_args", "{}")
    args.vis_encoder_model_path = getattr(args, "vis_encoder_model_path", None)
    args.vis_encoder_config_path = getattr(args, "vis_encoder_config_path", None)
    args.vis_encoder_finetune = getattr(args, "vis_encoder_finetune", False)
    args.vis_encoder_use_codebook = getattr(args, "vis_encoder_use_codebook", False)
    args.vis_encoder_embed_path = getattr(args, "vis_encoder_embed_path", None)
    args.vis_encoder_embed_dim = getattr(args, "vis_encoder_embed_dim", 128)
    args.vis_encoder_tokens = getattr(args, "vis_encoder_tokens", 8192)
    args.vis_encoder_grid_h = getattr(args, "vis_encoder_grid_h", 4)
    args.vis_encoder_grid_w = getattr(args, "vis_encoder_grid_w", args.vis_encoder_grid_h)
    args.vis_encoder_learned_pos = getattr(args, "vis_encoder_learned_pos", False)
    args.vis_encoder_use_2d_pos = getattr(args, "vis_encoder_use_2d_pos", 'sum')
    args.vis_encoder_hallucinate = getattr(args, "vis_encoder_hallucinate", 'none')
    args.vis_only = getattr(args, "vis_only", False)

    # vision-language learning args
    args.vislang_embed_dim = getattr(args, "vislang_embed_dim", args.encoder_embed_dim)
    args.vislang_embed_norm = getattr(args, "vislang_embed_norm", "none")
    args.halluc_model_path = getattr(args, "halluc_model_path", None)
    args.halluc_args = getattr(args, "halluc_args", "{}")
    args.mmt_inference = getattr(args, "mmt_inference", False)
    args.rand_inference = getattr(args, "rand_inference", False)
    args.halluc_only = getattr(args, "halluc_only", False)
    args.pretrain_mmt = getattr(args, "pretrain_mmt", None)


@register_model_architecture("SGHallucUMMT", "SGHallucUMMT_small")
def small_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    return base_architecture(args)


@register_model_architecture("SGHallucUMMT", "SGHallucUMMT_tiny")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    return base_architecture(args)
