import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    else:
        config = T5Config.from_pretrained("google-t5/t5-small")
        model = T5ForConditionalGeneration(config)
    
    model = model.to(DEVICE)
    return model

def freeze_layers(model, freeze_encoder_layers=None, freeze_decoder_layers=None, 
                  freeze_embeddings=False, freeze_lm_head=False):
    '''
    Freeze specific layers of the T5 model.
    
    Args:
        model: T5 model instance
        freeze_encoder_layers: List of encoder layer indices to freeze (0-indexed), 
                              or None to freeze all, or empty list to freeze none
        freeze_decoder_layers: List of decoder layer indices to freeze (0-indexed),
                              or None to freeze all, or empty list to freeze none
        freeze_embeddings: If True, freeze shared embeddings
        freeze_lm_head: If True, freeze the language model head
    
    Returns:
        model: Model with specified layers frozen
    '''
    num_encoder_layers = len(model.encoder.block)
    num_decoder_layers = len(model.decoder.block)
    
    # Freeze encoder layers
    if freeze_encoder_layers is not None:
        if freeze_encoder_layers == "all":
            # Freeze all encoder layers
            for i in range(num_encoder_layers):
                for param in model.encoder.block[i].parameters():
                    param.requires_grad = False
                print(f"Frozen encoder layer {i}")
        elif isinstance(freeze_encoder_layers, list) and len(freeze_encoder_layers) > 0:
            # Freeze specific encoder layers
            for i in range(num_encoder_layers):
                if i in freeze_encoder_layers:
                    for param in model.encoder.block[i].parameters():
                        param.requires_grad = False
                    print(f"Frozen encoder layer {i}")
    
    # Freeze decoder layers
    if freeze_decoder_layers is not None:
        if freeze_decoder_layers == "all":
            # Freeze all decoder layers
            for i in range(num_decoder_layers):
                for param in model.decoder.block[i].parameters():
                    param.requires_grad = False
                print(f"Frozen decoder layer {i}")
        elif isinstance(freeze_decoder_layers, list) and len(freeze_decoder_layers) > 0:
            # Freeze specific decoder layers
            for i in range(num_decoder_layers):
                if i in freeze_decoder_layers:
                    for param in model.decoder.block[i].parameters():
                        param.requires_grad = False
                    print(f"Frozen decoder layer {i}")
    
    # Freeze embeddings
    if freeze_embeddings:
        for param in model.shared.parameters():
            param.requires_grad = False
        print("Frozen shared embeddings")
    
    # Freeze LM head
    if freeze_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = False
        print("Frozen language model head")
    
    # Print summary of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nLayer Freezing Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)\n")
    
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    mkdir(checkpoint_dir)
    if best:
        path = os.path.join(checkpoint_dir, "best_model.pt")
    else:
        path = os.path.join(checkpoint_dir, "latest_model.pt")
    torch.save(model.state_dict(), path)

def load_model_from_checkpoint(args, best):
    model = initialize_model(args)
    if best:
        path = os.path.join(args.checkpoint_dir, "best_model.pt")
    else:
        path = os.path.join(args.checkpoint_dir, "latest_model.pt")
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded model from {path}")
    else:
        print(f"Warning: Checkpoint {path} not found. Using initialized model.")
    
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

