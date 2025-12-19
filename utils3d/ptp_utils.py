import math
from operator import index
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from diffusers.models.attention_processor import Attention
from utils3d.attn_utils import fn_smoothing_func

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    # print(query.shape, key.shape, value.shape)
    L, S = query.size(-2), key.size(-2) # 4429
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
    # if is_causal:
    #     assert attn_mask is None
    #     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    #     attn_bias.to(query.dtype)
    #
    # if attn_mask is not None:
    #     attn_mask = F.pad(attn_mask, (0, attn_bias.shape[1]-attn_mask.shape[1],
    #                0, attn_bias.shape[0]-attn_mask.shape[0]))
    #     if attn_mask.dtype == torch.bool:
    #         attn_bias += attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
    #     else:
    #         attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {}

    def __call__(self, attn, is_cross: str, place_in_unet: int):
        key = f"{place_in_unet}_{is_cross}"
        if self.cur_att_layer >= 0:
            self.step_store[key]=attn  # all mmdit attentions are of the same size in transformer
        self.cur_att_layer += 1
        # print(key,self.cur_att_layer,self.num_att_layers * self.attns_per_layer)
        if self.cur_att_layer == self.num_att_layers * self.attns_per_layer: # sort of attns, 4 for SD3 and 6 for SD35
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_attention(self):
        average_attention = self.attention_store
        return average_attention
    
    def feat_attention(self, from_where: List[int] = [6, 12], index:int = None ) -> torch.Tensor:
        attention_maps = self.get_attention()
        # consider img features
        sim_list = []
        for location in from_where:
            feat =  torch.mean(attention_maps[f"{location}_feat"], dim=0) # [4096, 64] average over heads
            norms_feat = [torch.linalg.norm(v)+1e-5 for v in feat]
            norms_feat = torch.stack(norms_feat)
            token =  torch.mean(attention_maps[f"{location}_token"], dim=0) # [20, 64] average over heads
            norms_token = [torch.linalg.norm(v)+1e-5 for v in token]
            norms_token = torch.stack(norms_token)
            sim = feat @ token.permute(1,0)
            sim = torch.clamp(sim, 0, 1)  # clamp due to possible negative values and extreme large
            result = torch.outer(norms_feat, norms_token)  # [4096, 20]
            sim = torch.div(sim, result)
            # sim = torch.clamp(sim, 0, 1)
            sim_list.append(sim)
        sim_list = torch.stack(sim_list, dim=0)
        sim_list = torch.mean(sim_list, dim=0)
        # print(torch.max(sim_list),torch.min(sim_list))
        # token_feat = torch.clamp(sim_list, 0, 1)
        # print(token_feat.shape) # [4096, 20#] patch-token

        attn = []
        for location in from_where:
            attn.append(attention_maps[f"{location}_feat"]) 
        attn = torch.stack(attn, dim=0)
        attn = torch.mean(attn, dim=(0,1))
        # print(attn.shape) # [4096, 64]

        return sim_list.reshape(64,64,-1).to(torch.float), attn.reshape(64,64,-1).to(torch.float)
    
    def aggregate_attention(self, from_where: List[int]=[5, 6, 7, 8, 9, 10, 11, 12], is_cross: bool = True, branch: bool = True) -> torch.Tensor:
        out = []
        attention_maps = self.get_attention()
        if is_cross:
            for location in from_where:
                values = torch.mean(attention_maps[f"{location}_cross"], dim=0) # max over heads
                attn_maps = values.reshape(64, 64, values.shape[-1]) # (64,64,333) or (64,64,4096)
                out.append(attn_maps)
            out = torch.stack(out, dim=0)
            average_over_layers = torch.mean(out, dim=0) # mean of all layers
            return average_over_layers.to(torch.float) #.cpu().detach()
        else:
            for location in from_where:
                values =  torch.mean(attention_maps[f"{location}_self"], dim=0)
                attn_maps = values.reshape(64, 64, values.shape[-1]) # (64,64,333) or (64,64,4096)
                out.append(attn_maps)
            out = torch.stack(out, dim=0)
            average_over_layers = torch.mean(out, dim=0) # mean of all layers
            return average_over_layers.to(torch.float) #.cpu().detach()
    
    def head_attention(self, from_where: List[int]=[5, 6, 7, 8, 9, 10, 11, 12], index:Union[List[int], int] = None, is_cross: bool = True) -> torch.Tensor:
        out = []
        attention_maps = self.get_attention()
        if is_cross:
            for location in from_where:
                temp = attention_maps[f"{location}_cross"][:,:,index].sum(dim=-1)
                # print(f'{location}_cross',[torch.trunc(temp[h].sum()).item() for h in range(temp.shape[0])])
                out.append(temp)
            out = torch.stack(out, dim=0)
            return out.to(torch.float) 
        else:
            for location in from_where:
                out.append(attention_maps[f"{location}_self"])
            out = torch.stack(out, dim=0)
            return out.to(torch.float) 

    def semantic_attention(self, from_where: List[int] = [5, 6, 7, 8, 9, 10, 11, 12], index:Union[List[int], int] = None, is_cross: bool = True) -> torch.Tensor:
        out = [] # for layer wise attention
        attention_maps = self.get_attention()
        if is_cross:
            for location in from_where:
                temp = attention_maps[f"{location}_cross"][:, :, index].sum(dim=-1)
                temp = torch.mean(temp, dim=0)
                out.append(temp)
            out = torch.stack(out, dim=0)
            return out.to(torch.float)
        else:
            for location in from_where:
                temp = torch.mean(attention_maps[f"{location}_self"], dim=0)
                out.append(temp)
            out = torch.stack(out, dim=0)
            return out.to(torch.float)

    def instance_attention(self, from_where: List[int]=[4,5,6,7], is_cross: bool = False, index:Union[List[int], int] = None) -> torch.Tensor:
        attn = []
        attention_maps = self.get_attention()
        if is_cross:
            for location in from_where:
                values,_ = torch.max(attention_maps[f"{location}_cross"][:,:,index].sum(dim=-1), dim=0)
                attn_maps = values.reshape(64, 64)
                attn.append(attn_maps)
            out = torch.stack(attn, dim=0)
            out,_ = torch.max(out, dim=0)
            return out.to(torch.float)  # .cpu().detach()
        else:
            # from_where = {6: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]}
            from_where = {6: [1, 8, 10, 17], 7: [4, 11], 8: [1, 2, 7, 8, 9, 18, 20], 10: [5, 8, 9], 11: [4, 15, 18], 12: [0, 22]}  # SD3
            for location in from_where.keys():
                for i in from_where[location]:
                    attn.append(attention_maps[f"{location}_self"][i])
            attn = torch.stack(attn, dim=0)
            attn = torch.mean(attn, dim=0)  # average over layers and heads
            attn = attn.reshape(64, 64, -1).to(torch.float)
        return attn

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self,attns_per_layer:int=None):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attns_per_layer=attns_per_layer

class AttnProcessor:
    def __init__(self, model_choice, attnstore, place_in_transformer, from_where, attention_mask):
        super().__init__()
        self.model_choice = model_choice,
        self.attnstore = attnstore
        self.place_in_transformer = place_in_transformer
        self.from_where=from_where
        self.attention_mask=attention_mask

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            *args,
            **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections. encoder_hidden_states is text tokens embedding.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            image_length = query.shape[2] - encoder_hidden_states_query_proj.shape[2]

        ####################################################################################################
        if query.requires_grad and encoder_hidden_states is not None and self.place_in_transformer in self.from_where:
            hidden_states, attention_probs = scaled_dot_product_attention(query, key, value, attn_mask=self.attention_mask, dropout_p=0.0, is_causal=False)
            # (1,24,4429,4429) -> (1,24,4096,333) -> (1,4096,333)
            # to save space we only take part of them
            cross_attn_clip1 = attention_probs[:, :, :image_length, image_length:image_length + 77]  # .cpu() image-text cross attention clip
            cross_attn_clip2 = attention_probs[:, :, :image_length, image_length + 77:image_length + 154]  # .cpu() image-text cross attention t5  image_length + 97
            cross_attn = torch.squeeze(torch.maximum(cross_attn_clip1, cross_attn_clip2))
            self.attnstore(cross_attn, 'cross', self.place_in_transformer)
            # (1,24,4429,4429) -> (1,24,4096,4096) -> (1,4096,4096)
            self_attn = attention_probs[:, :, :image_length, :image_length]  # .cpu() image-image self attention
            # if self.model_choice == ("stabilityai/stable-diffusion-3-medium-diffusers",):
            self.attnstore(torch.squeeze(self_attn), 'self', self.place_in_transformer)
            
            # if self.place_in_transformer>12:
            #     # print(self.place_in_transformer)
            #     self.attnstore(torch.squeeze(self_attn), 'self2', self.place_in_transformer)

            feat = hidden_states[:, :, :image_length, :]
            token = hidden_states[:, :, image_length:, :]  # image_length + 20
            self.attnstore(torch.squeeze(feat), 'feat', self.place_in_transformer)
            self.attnstore(torch.squeeze(token), 'token', self.place_in_transformer)
        # elif query.requires_grad and encoder_hidden_states is None and self.place_in_transformer in self.from_where:  # this is for SD3.5m
        #     hidden_states, attention_probs = scaled_dot_product_attention(query, key, value, attn_mask=self.attention_mask, dropout_p=0.0, is_causal=False)
        #     self.attnstore(torch.squeeze(attention_probs), 'self2', self.place_in_transformer)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        ####################################################################################################

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states



class HeadsAttentionStore:
    @staticmethod
    def get_empty_store():
        return {}

    def __call__(self, attn, is_cross: str, place_in_unet: int):
        key = f"{place_in_unet}_{is_cross}" # 3 possible kinds of keys
        if self.cur_att_layer >= 0:
            self.step_store[key] = attn  # all mmdit attentions are of the same size in transformer
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers *2: # 2 possible kinds of keys from features
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_attention(self):
        average_attention = self.attention_store
        return average_attention
    
    def aggregate_attention(self, from_where: List[int] = [5,6,7,8,9,10,11,12], is_cross: bool = True) -> torch.Tensor:
        out = []
        attention_maps = self.get_attention()
        if is_cross:
            sim_list = []
            for location in from_where:
                feat =  torch.mean(attention_maps[f"{location}_feat"], dim=0) # [4096, 64] average over heads
                norms_feat = [torch.linalg.norm(v) for v in feat]
                norms_feat = torch.stack(norms_feat)
                token =  torch.mean(attention_maps[f"{location}_token"], dim=0) # [20, 64] average over heads
                norms_token = [torch.linalg.norm(v) for v in token]
                norms_token = torch.stack(norms_token)
                sim = feat @ token.permute(1,0)
                result = torch.outer(norms_feat, norms_token)  # [4096, 20]
                sim = torch.div(sim, result)
                # sim = sim[:, 1:-1]
                # sim = sim * 100
                # sim = torch.nn.functional.softmax(sim, dim=-1)
                sim_list.append(sim)  # [:,index]
            sim_list = torch.stack(sim_list, dim=0)
            sim_list = torch.mean(sim_list, dim=0)
            # print(torch.max(sim_list),torch.min(sim_list))
            token_feat = torch.clamp(sim_list, 0, 1)  #[4096]
            return token_feat.to(torch.float).reshape(64,64,-1)
        else:
            # consider img-img features
            feat_list = []
            for location in from_where:
                feat_list.append(attention_maps[f"{location}_feat"])
            feat_list = torch.stack(feat_list, dim=0)
            feat_list = feat_list.permute(2,0,1,3)
            feat_list = feat_list.reshape(feat_list.shape[0],-1)
            # print(feat_list.shape) # [4096, L*24*64]
            size = feat_list.shape[0]
            feat_sim = feat_list @ feat_list.permute(1,0)
            norms1 = [torch.linalg.norm(v) for v in feat_list]
            norms1 = torch.stack(norms1)
            result = torch.outer(norms1, norms1)
            feat_sim = torch.div(feat_sim, result)/ math.sqrt(norms1.shape[0])  # scale
            feat_sim = torch.clamp(feat_sim, 0, 1)
            # print(feat_sim.shape)  # [4096, 4096]
            return feat_sim.to(torch.float).reshape(64,64,-1)

    def feat_token(self, from_where: List[int] = [5,6,7,8,9,10,11,12], from_where2: List[int] = [5,6,7,8,9,10,11,12],index:int = None ) -> torch.Tensor:
        attention_maps = self.get_attention()
        # consider text-img features
        sim_list = []
        index = [index[id]-1 for id in range(len(index))]
        for location in from_where:
            feat =  torch.mean(attention_maps[f"{location}_feat"], dim=0) # [4096, 64] average over heads
            norms_feat = [torch.linalg.norm(v) for v in feat]
            norms_feat = torch.stack(norms_feat)
            token =  torch.mean(attention_maps[f"{location}_token"], dim=0) # [20, 64] average over heads
            norms_token = [torch.linalg.norm(v) for v in token]
            norms_token = torch.stack(norms_token)
            sim = feat @ token.permute(1,0)
            result = torch.outer(norms_feat, norms_token)  # [4096, 20]
            sim = torch.div(sim, result)

            sim = sim[:, 1:-1]
            sim = sim * 100
            sim = torch.nn.functional.softmax(sim, dim=-1)

            sim_list.append(sim[:,index])
        sim_list = torch.stack(sim_list, dim=0)
        sim_list = torch.mean(sim_list, dim=0)
        # print(torch.max(sim_list),torch.min(sim_list))
        token_feat = torch.clamp(sim_list, 0, 1)  #[4096]

        # consider img-img features
        feat_list = []
        for location in from_where2:
            feat_list.append(attention_maps[f"{location}_feat"])
        feat_list = torch.stack(feat_list, dim=0)
        feat_list = feat_list.permute(2,0,1,3)
        feat_list = feat_list.reshape(feat_list.shape[0],-1)
        # print(feat_list.shape) # [4096, L*24*64]
        size = feat_list.shape[0]
        feat_sim = feat_list @ feat_list.permute(1,0)
        norms1 = [torch.linalg.norm(v) for v in feat_list]
        norms1 = torch.stack(norms1)
        result = torch.outer(norms1, norms1)
        feat_sim = torch.div(feat_sim, result)
        feat_sim = torch.clamp(feat_sim, 0, 1)
        # print(feat_sim.shape)  # [4096, 4096]

        tp_sim = feat_sim @ token_feat
        return token_feat.to(torch.float).reshape(64,64), feat_sim.to(torch.float).reshape(64,64,-1), tp_sim.to(torch.float).reshape(64,64)
  
    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0




class HeadAttnProcessor:
    def __init__(self, attnstore, place_in_transformer, from_where, attention_mask):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_transformer = place_in_transformer
        self.from_where=from_where
        self.attention_mask=attention_mask

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            *args,
            **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]
        # `sample` projections.
        # print(hidden_states.shape)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        # print(query.shape)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            image_length = query.shape[2] - encoder_hidden_states_query_proj.shape[2]
            # print(query.shape, key.shape, value.shape)

        ####################################################################################################
        # if encoder_hidden_states is not None and query.requires_grad and self.place_in_transformer in self.from_where:
            
        # else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=self.attention_mask, dropout_p=0.0, is_causal=False)
            feat = hidden_states[:, :, :image_length, :]
            token = hidden_states[:, :, image_length:image_length + 20, :]
            self.attnstore(torch.squeeze(feat), 'feat', self.place_in_transformer)
            self.attnstore(torch.squeeze(token), 'token', self.place_in_transformer)
        ####################################################################################################

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states