"""
Script to evaluate all experiments on test2 set.
This will run inference on test2.nl and evaluate against test2.sql for all your experiments.
"""

import os
import argparse
from tqdm import tqdm

import torch
from transformers import T5TokenizerFast

from t5_utils import initialize_model, load_model_from_checkpoint
from load_data import get_dataloader
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate all experiments on test2 set')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                       help='List of experiment names to evaluate (e.g., baseline experiment). If not provided, evaluates all found experiments.')
    parser.add_argument('--test_batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Base directory for checkpoints')
    args = parser.parse_args()
    return args

def test2_inference(model, test2_loader, model_sql_path, model_record_path):
    """Run inference on test2 set."""
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    
    decoder_start_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if decoder_start_token_id == tokenizer.unk_token_id:
        decoder_start_token_id = tokenizer.pad_token_id
    
    all_generated_sql = []
    
    with torch.no_grad():
        for batch in tqdm(test2_loader, desc="Generating test2 predictions"):
            encoder_input, encoder_mask, initial_decoder_inputs = batch
            
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=2,
                decoder_start_token_id=decoder_start_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_sql = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated_sql.extend(generated_sql)
    
    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)
    return all_generated_sql

def find_experiments(base_dir='checkpoints'):
    """Find all experiment directories."""
    experiments = []
    if not os.path.exists(base_dir):
        return experiments
    
    for model_type in ['ft_experiments', 'scr_experiments']:
        type_dir = os.path.join(base_dir, model_type)
        if os.path.exists(type_dir):
            for exp_name in os.listdir(type_dir):
                exp_path = os.path.join(type_dir, exp_name)
                if os.path.isdir(exp_path):
                    best_model = os.path.join(exp_path, 'best_model.pt')
                    if os.path.exists(best_model):
                        experiments.append((model_type, exp_name))
    
    return experiments

def main():
    args = get_args()
    
    # Load test2 data
    print("Loading test2 data...")
    test2_loader = get_dataloader(args.test_batch_size, "test2")
    
    # Find experiments
    if args.experiments:
        # User specified experiments - need to determine model type
        experiments = []
        for exp_name in args.experiments:
            # Try to find in both ft and scr
            for model_type in ['ft_experiments', 'scr_experiments']:
                exp_path = os.path.join(args.checkpoint_dir, model_type, exp_name)
                best_model = os.path.join(exp_path, 'best_model.pt')
                if os.path.exists(best_model):
                    experiments.append((model_type, exp_name))
                    break
    else:
        # Find all experiments
        experiments = find_experiments(args.checkpoint_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"\nFound {len(experiments)} experiment(s) to evaluate:")
    for model_type, exp_name in experiments:
        print(f"  - {model_type}/{exp_name}")
    
    # Ground truth paths
    gt_sql_path = 'data/test2.sql'
    gt_record_path = 'records/ground_truth_test2.pkl'
    
    # Create ground truth records if they don't exist
    if not os.path.exists(gt_record_path):
        print("\nCreating ground truth records for test2...")
        from utils import compute_records
        import pickle
        
        with open(gt_sql_path, 'r') as f:
            gt_sql_queries = [line.strip() for line in f.readlines()]
        
        records, error_msgs = compute_records(gt_sql_queries)
        os.makedirs('records', exist_ok=True)
        with open(gt_record_path, 'wb') as f:
            pickle.dump((records, error_msgs), f)
        print(f"Created {gt_record_path}")
    
    # Evaluate each experiment
    results = []
    for model_type, exp_name in experiments:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_type}/{exp_name}")
        print(f"{'='*80}")
        
        # Determine if finetune or scratch
        finetune = 'ft' in model_type
        model_type_short = 'ft' if finetune else 'scr'
        
        # Create args object for model loading
        class Args:
            def __init__(self):
                self.finetune = finetune
                self.checkpoint_dir = os.path.join(args.checkpoint_dir, model_type, exp_name)
        
        model_args = Args()
        
        # Load model
        model = load_model_from_checkpoint(model_args, best=True)
        model.eval()
        
        # Run inference
        model_sql_path = f'results/t5_{model_type_short}_{exp_name}_test2.sql'
        model_record_path = f'records/t5_{model_type_short}_{exp_name}_test2.pkl'
        
        print("Running inference on test2...")
        test2_inference(model, test2_loader, model_sql_path, model_record_path)
        
        # Evaluate
        print("Computing metrics...")
        sql_em, record_em, record_f1, error_msgs = compute_metrics(
            gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        
        error_rate = sum(1 for msg in error_msgs if msg != "") / len(error_msgs) if error_msgs else 0
        
        results.append({
            'experiment': exp_name,
            'model_type': model_type_short,
            'sql_em': sql_em,
            'record_em': record_em,
            'record_f1': record_f1,
            'error_rate': error_rate
        })
        
        print(f"\nResults for {exp_name}:")
        print(f"  SQL Exact Match: {sql_em:.4f}")
        print(f"  Record Exact Match: {record_em:.4f}")
        print(f"  Record F1: {record_f1:.4f}")
        print(f"  SQL Error Rate: {error_rate*100:.2f}%")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - Test2 Evaluation Results")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'SQL EM':<10} {'Record EM':<12} {'Record F1':<12} {'Error %':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['experiment']:<30} {r['sql_em']:<10.4f} {r['record_em']:<12.4f} {r['record_f1']:<12.4f} {r['error_rate']*100:<10.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()

