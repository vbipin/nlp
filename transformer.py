from __future__ import unicode_literals, print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#import transformer.Constants as Constants

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def position_encoding_init(max_len, d_model):
    #from: http://nlp.seas.harvard.edu/2018/04/03/attention.html
    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position  = torch.tensor( torch.arange(0, max_len), dtype=torch.float ).unsqueeze(1)
    all_range = torch.tensor( torch.arange(0, d_model, 2), dtype=torch.float )
    #div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    div_term = torch.exp(all_range * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    #print (pe.shape)
    return pe


        
#We find the input embeddings of the incoming index and add the positional encodings
#This class works just like the nn.Embedding class
#Input: LongTensor of arbitrary shape containing the indices to extract
#Output: (*, embedding_dim), where * is the input shape
class TransformerEmbedding( nn.Module ) :
    def __init__( self, n_vocab, n_position, d_model ) :
        super(TransformerEmbedding, self).__init__()
        
        self.positional_enc = nn.Embedding(n_position, d_model).from_pretrained( position_encoding_init(n_position, d_model ) )
        self.input_emb      = nn.Embedding(n_vocab, d_model )
     
    def forward( self, input_seq, input_positions ) :
        """
        input is of shape (seq, batch)
        output is of shape (batch, seq, emb)        
        """
        #Embedding layer of input + positional encoding is the input 
        embeddings = self.input_emb( input_seq ) + self.positional_enc( input_positions ) 
        #remember this is of shape (seq, batch, emb)
        #we change it to (batch, seq, enb ) for further processing.
        return embeddings #.transpose(0,1) #(batch, seq, emb)
        
 
    
#ref: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
def get_padding_mask(seq, expand): #seq: (batch, seq_len )
    ''' Indicate the padding-related part to mask '''
    batch, seq_len = seq.shape
    pad_mask = seq == PAD #1 at all positions where PAD is, and 0 for notPAD.
    pad_mask = pad_mask.unsqueeze(1).expand(batch, expand, seq_len) # b, expand, seq_len
    return pad_mask

def get_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask



#################################################
#Scaled DotProduct attention
#Initial code is mine; changed the indexan dmatmul funtions from here:http://nlp.seas.harvard.edu/2018/04/03/attention.html
def scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout=None ) : 
    """As given in the paper
        Q, K, V are matrices
        Q    shape => ( batch, n_queries, d_q )
        K, V shape => ( batch, n_sequence, d_v )

        Output is the vector and attn scores
        out  => (batch, n_queries, d_v)
        attn => (batch, n_queries, n_sequence)
    """
    #print(Q.shape, K.shape, V.shape )
    #This is the attention function. Here it is just dot product.
    scores = torch.matmul( Q, K.transpose(-2,-1) ) / np.sqrt( K.shape[-1] ) 

    #Before softmax we need to apply the mask.
    #mask is just to invalidate certain entries int the scores
    if attn_mask is not None :
        #print(Q.shape, K.shape, V.shape )  #XXX debug
        #print(attn_mask.size(), scores.size())
        assert attn_mask.size() == scores.size() 
        #wherever the mask is set as 1, fill the scores with -inf so that softmax values will be zeros.
        #masked_fill_(mask,value): Fills elements of self tensor with value where mask is one.
        scores.data.masked_fill_(attn_mask, -1e9) #-np.inf?

    attn = F.softmax( scores, dim=-1 ) #-1 is the last dim
    #print(scores.shape)

    if dropout is not None :
        attn = dropout(attn)

    out = torch.matmul( attn, V )

    return out, attn 
    
#################################################
#Multi head Attention

#Multi head has h different heads
#one such head is SingleHeadAttention

class SingleHeadAttention(nn.Module) :
    def __init__( self, d_model, d_k, d_v  ) : #d_model == dk in the paper
        """ Input are d_model, d_k, and d_v"""
        super(SingleHeadAttention, self).__init__()
        
        self.wq = nn.Linear(d_model, d_k)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_v)
        
    def forward(self, Q, K, V, attn_mask=None ) : 
        """
        Q    shape => ( batch, n_queries, d_q )
        K, V shape => ( batch, n_sequence, d_v )
        """
        out, attn_scores = scaled_dot_product_attention( self.wq(Q), self.wk(K), self.wv(V), attn_mask)
        #out, attn_scores = debug_attention( self.wq(Q), self.wk(K), self.wv(V), attn_mask ) 
        self.attn_scores = attn_scores
        return out #(batch, n_queries, d_v)
    
    
class MultiHeadAttention(nn.Module) :

    def __init__( self, d_model, h, d_k, d_v  ) : #h is the number of heads, d_model == dk in the paper
        """ Input are d_model, number of heads h, d_k, and d_v"""
        super(MultiHeadAttention, self).__init__()
        
        self.multi_attn = nn.ModuleList( [ SingleHeadAttention(d_model, d_k, d_v) for _ in range(h) ] )              
        self.out_linear = nn.Linear( h*d_v, d_model )
        
    def forward( self, Q, K, V, attn_mask=None ) :
        """As given in the paper
        Q, K, V are matrices
        Q    shape => ( batch, n_queries, d_q )
        K, V shape => ( batch, n_sequence, d_v )

        Output is the vector and attn scores
        out  => (batch, n_queries, d_model)
        attn => [(batch, n_queries, n_sequence) ... for each head ]
        """
        
        heads =  [ a(Q,K,V,attn_mask) for a in self.multi_attn ] #returns a tuple of head and attns
        
        concat_head = torch.cat( heads, dim=-1 ) #we concat the last dim of d_v size vectors to get h*d_v.  
        self.debug_heads = heads #XXX
        out = self.out_linear( concat_head )
        
        self.debug_multi_head_out = out #XXX
        
        return out   #(batch, n_queries, d_model)
        
        
class PositionWiseFFN( nn.Module ) : #position wise ffed forward network    
    def __init__( self, d_model, d_hidden ) :
        super(PositionWiseFFN, self).__init__()        
        self.w1 = nn.Linear( d_model, d_hidden )
        #there will be a relu inbetween
        self.w2 = nn.Linear( d_hidden, d_model )

    def forward( self, x ) :
        """
        input  (batch, seq, d_model)
        output (batch, seq, d_model)
        """
        out = self.w2( F.relu( self.w1(x) ))
        return out #(batch, seq, d_model)


        


        
#The transformer Encoder layer with multihead attention and FFN
#This layer is stacked N times in the encoder. N = 6
class TransformerEncoderLayer( nn.Module ) :
    def __init__( self, d_model, n_head, d_k, d_v, d_ffn ) :
        """
        Encoder Layer has a multi head attention sub layer and a feed forward network layer
        The sublayers have residulal connections. Output from each sublayer is LayerNormalized
        """
        
        super(TransformerEncoderLayer, self).__init__()
        
        self.multi_head_attn = MultiHeadAttention( d_model, n_head, d_k, d_v )
        #self.multi_head_attn = debug_MultiHeadedAttention(n_head, d_model) #XXX debug
        self.ffn = PositionWiseFFN( d_model, d_ffn )
        
        #self.norm1 = nn.LayerNorm( d_model )
        #self.norm2 = nn.LayerNorm( d_model )
        
        self.norm1 = nn.LayerNorm( d_model )
        self.norm2 = nn.LayerNorm( d_model )
        
    def forward( self, x, attn_mask=None ) :
        
        residual = x 
        #print(input.shape)
        out = self.multi_head_attn( x, x, x, attn_mask )
        self.debug_attn_out = out #XXX
        
        out += residual #add the residual connection        
        ffn_input = self.norm1( out )
        self.debug_ffn_input = ffn_input #XXX
        
        layer_out = ffn_input #XXX for debug. remove the FFN layer
        """
        residual = ffn_input
        ffn_out = self.ffn( ffn_input )
        self.debug_ffn_out = ffn_out #XXX
        
        ffn_out += residual
        layer_out = self.norm2( ffn_out )
        """
        self.debug_layer_out = layer_out  #XXX        
        return layer_out #, attn 
    
#Transformer Encoder 
class TransformerEncoder( nn.Module ) :
    
    def __init__( self, n_src_vocab, n_max_seq, n_layers=6, d_model=512, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_ffn=1024, dropout=0.1) :
        
        super(TransformerEncoder, self).__init__()
        
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        
        self.hidden_size = d_model #XXX just for compatibility with other encoders
        
        #for processing the input
        self.input_emb = TransformerEmbedding( n_src_vocab, n_position, d_model )
        
        #N layers of transformer
        #self.layers = TransformerNLayers( n_layers, d_model, n_head, d_k, d_v, d_ffn )
        self.layer_list = nn.ModuleList([ TransformerEncoderLayer( d_model, n_head, d_k, d_v, d_ffn ) for _ in range(n_layers) ])
        #self.layers = nn.Sequential( *self.layer_list )
        
    def forward( self, input_seq, input_positions) :
        """
        input  (batch, seq),   positions (batch, seq)
        output (batch, seq, d_model)
        """
        #Embedding layer of input + positional encoding is the input
        emb_input = self.input_emb( input_seq, input_positions )
        attn_mask = get_padding_mask(input_seq, input_seq.shape[1]) #(batch, seq_len, seq_len )
        
        # pipe it through all the N layers
        enc_output = emb_input #init variable
        for layer in self.layer_list :
            enc_output = layer.forward( enc_output, attn_mask )
        #enc_output = self.layers( emb_input )
        
        return enc_output #(batch, seq, d_model)
    
    
#The transformer Decoder layer with multihead attention and FFN
#This layer is stacked N times in the encoder. N = 6
class TransformerDecoderLayer( nn.Module ) :
    def __init__( self, d_model, n_head, d_k, d_v, d_hidden ) :
        """
        Decoder Layer has a 
            masked multi head self attention sublayer, 
            multi head attention sub layer 
            and a feed forward network layer
        The sublayers have residulal connections. Output from each sublayer is LayerNormalized
        """
        super(TransformerDecoderLayer, self).__init__()
        
        self.masked_multi_head_attn = MultiHeadAttention( d_model, n_head, d_k, d_v )
        self.multi_head_attn = MultiHeadAttention( d_model, n_head, d_k, d_v )
        self.ffn = PositionWiseFFN( d_model, d_hidden )
        
        self.norm1 = nn.LayerNorm( d_model )
        self.norm2 = nn.LayerNorm( d_model )
        self.norm3 = nn.LayerNorm( d_model )
        
    def forward( self, input, input_mask, encoder_output, enc_mask  ) : #for nn.Sequential to work we give only one argument.
        #print("debug")
        x = input
        residual = x 
        #print(input.shape)
        out = self.masked_multi_head_attn( x, x, x, attn_mask=input_mask )
        out += residual #add the residual connection        
        next_input = self.norm1( out )
        
        residual = next_input 
        #print(input.shape)
        out = self.multi_head_attn( next_input, encoder_output, encoder_output, attn_mask=enc_mask )
        out += residual #add the residual connection        
        ffn_input = self.norm2( out )
        
        residual = ffn_input
        ffn_out = self.ffn( ffn_input )
        ffn_out += residual
        layer_out = self.norm3( ffn_out )
                
        return layer_out #XXX Mask Not correct 
    
      
#Transformer Encoder 
class TransformerDecoder( nn.Module ) :
    
    def __init__( self, n_tgt_vocab, n_max_seq, n_layers=6, d_model=512, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_ffn=1024, dropout=0.1) :
        
        super(TransformerDecoder, self).__init__()
        
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        
        self.hidden_size = d_model #XXX just for compatibility with other encoders
        
        #for processing the input
        self.input_emb = TransformerEmbedding( n_tgt_vocab, n_position, d_model )
        
        #N layers of transformer
        self.layer_list = nn.ModuleList([ TransformerDecoderLayer( d_model, n_head, d_k, d_v, d_ffn ) for _ in range(n_layers) ])    
        #self.layers = nn.Sequential( *self.layer_list )
        
    def forward( self, input_seq, input_positions, dec_mask, enc_output, enc_mask ) :
        """
        input  (batch, seq),   positions (batch, seq)
        output (batch, seq, d_model)
        """
        #Embedding layer of input + positional encoding is the input
        dec_input = self.input_emb( input_seq, input_positions )
        #print(input_seq.shape)
        #print(dec_input.shape)
        # pipe it through all the N layers
        dec_output = dec_input
        for layer in self.layer_list : #
            dec_output = layer.forward(dec_output, dec_mask, enc_output, enc_mask) 
        
        self.debug2 = dec_output.shape
        self.debug3 = dec_output
        
        return dec_output #(batch, seq, d_model)       

#ref: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, 
                    n_src_vocab, n_tgt_vocab, n_max_seq, 
                    n_layers=6, n_head=8, d_word_vec=512, d_model=512, d_ffn=1024, d_k=64, d_v=64,
                    dropout=0.1):

        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_ffn=d_ffn, dropout=dropout)
        
        self.decoder = TransformerDecoder(
            n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_ffn=d_ffn, dropout=dropout)
        
        self.tgt_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module output shall be the same.'


    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos ):
        #src_seq, src_pos = src
        #tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]
        
        dec_slf_attn_pad_mask = get_padding_mask(tgt_seq, tgt_seq.shape[1] ) #(batch,seq,seq)
        
        dec_slf_attn_sub_mask = get_subsequent_mask(tgt_seq) #(batch,seq,seq)
        
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_pad_mask = get_padding_mask(src_seq, tgt_seq.shape[1]) #(batch, tgt_seq_len, src_seq_len )
        

        enc_output = self.encoder(src_seq, src_pos)
        dec_output = self.decoder(tgt_seq, tgt_pos, dec_slf_attn_mask, enc_output, dec_enc_attn_pad_mask )
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))
    
    
    