#In this file we keep the sequence to sequence encoders and decoders that are used for NMT

#this encoder is a simple one. It just adds all the embeeded word vectors to a single vector.


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderSimple(nn.Module):
    def __init__(self, src_vocab_size, hidden_size ):
        super(EncoderSimple, self).__init__()
        self.hidden_size = hidden_size
        
        #embedding vector size is fixed as hidden size
        self.enbedding_vector_size = hidden_size
        self.embedding = nn.Embedding(src_vocab_size, self.enbedding_vector_size )        

    def forward(self, input ):
        embedded = self.embedding(input).view(-1)
        self.embedded = embedded.view( input.shape[0], 1, -1 ) #seq_length, batch, enbbding
        self.out = torch.sum(self.embedded,dim=0)
        return self.out
    
class EncoderRNN(nn.Module):
    def __init__(self, src_vocab_size, hidden_size, num_layers=1 ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        #embedding vector size is fixed as hidden size
        self.enbedding_vector_size = hidden_size
        self.embedding = nn.Embedding(src_vocab_size, self.enbedding_vector_size )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1)
        output = embedded.view( input.shape[0], 1, -1 ) #seq_length, batch, enbbding
        #print (output.shape)
        #print (hidden.shape)
        self.output, self.hidden = self.gru(output, hidden)
        return self.output, self.hidden

    def init_hidden(self):
        result = torch.zeros(1, 1, self.hidden_size).to(device)     
        return result
        
        
#GRU based decoder
#input is one at a time.
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, dest_vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        #embedding vector size is fixed as hidden size
        self.enbedding_vector_size = hidden_size
        self.embedding = nn.Embedding(dest_vocab_size, self.enbedding_vector_size )
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.linear = nn.Linear(hidden_size, dest_vocab_size)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1)
        #output = F.relu(output)
        output = embedded.view( input.shape[0], 1, -1 ) #input shape[0] is 1 as wqe feed one input at a time.
        
        output, hidden = self.gru(output, hidden)
        output = self.linear( output.squeeze() )
        #print (output.shape)
        output = F.log_softmax( output, dim=0 )
        return output.view(1,-1), hidden #output of shape N,C; here N=1

    def initHidden(self):
        result = torch.zeros(1, 1, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class Attn(nn.Module) :
    def __init__(self, hidden_size, max_length) :
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.linear = nn.Linear(self.hidden_size, self.max_length)


    def forward(self, hidden, encoder_outputs) :

        attn_scores = self.linear(hidden)
        #print("attn_scores", attn_scores.shape )

        attn_weights = F.softmax(attn_scores, dim=2)

        #print("attn_weights", attn_weights.shape)
        #print("encoder_outputs",encoder_outputs.shape)

        attn_applied = torch.matmul(attn_weights.squeeze(),encoder_outputs)
        #print ("attn_applied ", attn_applied.shape)

        return attn_applied, attn_weights
        
class AttnDecoderRNN(nn.Module):   
    def __init__(self, hidden_size, output_size, max_length ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        
        #embedding vector size is fixed as hidden size
        self.enbedding_vector_size = hidden_size

        self.embedding = nn.Embedding(self.output_size, self.enbedding_vector_size)
        
        self.attn = Attn(self.hidden_size, self.max_length)
        #self.attn = nn.Linear(self.hidden_size, self.max_length)        
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        #self.dropout = nn.Dropout(self.dropout_p)
        
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """input is an index of the word. We create a word vector out of it"""
        embedded = self.embedding(input) 
        #print("embedded", embedded.shape )
                
        """ gru hidden has shape (num_layers * num_dir, batch, hidden_size)
            Here first two dim are 1
        """
        output, hidden = self.gru(embedded.view(1,1,-1), hidden)
        #print ("hidden ", hidden.shape)
        
        #linear W.h 
        #out (max, )
        attn_context, attn_weights = self.attn( hidden, encoder_outputs)
        #print ("attn_context ", attn_context.shape)
        
        
        output = torch.cat((hidden.view(1,-1), attn_context.view(1,-1)), 1)
        #print ("output ", output.shape) 
        
        output = self.attn_combine(output)
        #print ("output ", output.shape)        
        output = F.relu(output) #h tilde
        #print ("output ", output.shape)
        
        #output = F.log_softmax(self.out(output), dim=1)
        output = self.out(output)
        #print ("output ", output.shape)
        
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        

        
#EncodeDecode run
#still in development. Take from pytorch_nmt_encode-decode notebook
#this is just for backup

#17/7/18
class ___EncodeDecodeRun :
    def __init__(self, encoder, decoder) :
        self.encoder = encoder
        self.decoder = decoder
        
        #these values will be filled by the encoders.
        self.g = None
        self.attn_values = None
        self.out = None
        
        if isinstance(encoder, EncoderSimple) :
            self.run_encoder = self.run_simple_encoder
        
        if isinstance(encoder, EncoderRNN) :
            self.run_encoder = self.run_rnn_encoder
            
        if isinstance(decoder, DecoderRNN) :
            self.run_decoder = self.run_rnn_decoder
            
        if isinstance(decoder, AttnDecoderRNN) :
            self.run_decoder = self.run_attn_decoder 
    
    def run_simple_encoder(self, x ) :
        self.out = self.encoder(x)
        self.g = self.out.view(1,1,-1)
        self.attn_values = self.encoder.embedded
        
        
    def run_rnn_encoder(self, x ) :
        self.g = self.encoder.init_hidden()
        
        self.out, self.g = self.encoder(x, self.g )
        self.g.detach_()
        self.attn_values = self.out
        
        
    def run_rnn_decoder(self, yi ) :
            #for i in range(1) :
            scores, self.g = self.decoder( yi, self.g)
            #print(scores.shape)
            #print(next_word.shape)
            return scores
    
    def run_attn_decoder(self, yi ) :
        #self.attn_values is of shape (n,d)
        #we need it as (MAX_LENGTH, d) witht he first n filled
        max_length = self.decoder.max_length
        values_to_attend = torch.zeros(max_length, self.decoder.hidden_size, device=device)
        for i in range(self.attn_values.shape[0]) :
            values_to_attend[i] = self.attn_values[i][0]
                
        scores, self.g, _ = decoder( yi, self.g, values_to_attend )
        #print(scores.shape)
        #print(next_word.shape)
        return scores

def ___train(encdecrun, encoder_optimizer, decoder_optimizer, criterion, train_iter, n_data=5000 ) :
    start = time()
    
    loss_db = []
    for x, y in train_iter :
        x = x.to(device)
        y = y.to(device)
        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        #h = encoder.initHidden().to(device)
        #h.detach_()
        encdecrun.run_encoder( x )
    
        #g = torch.sum( encoder_outputs, dim=0 ).view(1,1,-1)
        
        y = y.detach()
        y_len = y.shape[0] #size of sequence
        for i in range(y_len - 1) :
            scores = encdecrun.run_decoder( y[i] )
            loss += criterion(scores, y[i+1] )

        loss.backward()
        loss_db.append( float(loss) )
                
        encoder_optimizer.step()
        decoder_optimizer.step()
        if n_data < 0 :
            break
        else :
            n_data -= 1
        
    end = time()
    print (end-start)
    return loss_db