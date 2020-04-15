from datetime import datetime

class Parameters:
    def __init__(self):
        self.n_epochs           = 1
        self.batch_size         = 32
        self.n_updates          = 2049
        self.learning_rate      = 4e-3
        self.log_interval       = 64
        self.dataset            = 'CIFAR10'
        
        self.encode             = 'basic'
        self.in_dim             = 1
        self.h_dim              = 32
        self.res_h_dim          = 128
        self.n_res_layers       = 32

        # whether or not to save model
        self.save       = True
        self.filename   = 'test1'
        self.print_interval = 16
        
        self.device = 'cuda'

        self.n_hiddens = 32

        # laskin
        self.n_residual_hiddens = 32
        self.n_residual_layers  = 2
        self.embedding_dim      = 64
        self.n_embeddings       = 512
        self.beta               = .25