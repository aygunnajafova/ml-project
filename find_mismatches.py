#!/usr/bin/env python3
"""
Find, print, and save mismatched queries between ground truth and model predictions.

Usage:
    python find_mismatches.py --experiment_name experiment --finetune
    python find_mismatches.py --experiment_name exp1 --model_type ft
"""

import os
import argparse
import pickle
from utils import read_queries

def main():
    parser = argparse.ArgumentParser(description='Find mismatched queries between ground truth and model predictions')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name (same as used in train_t5.py)')
    parser.add_argument('--finetune', action='store_true',
                        help='Whether model was fine-tuned (determines model_type as ft)')
    parser.add_argument('--model_type', type=str, choices=['ft', 'scr'], default=None,
                        help='Model type: ft (fine-tuned) or scr (scratch). If not provided, inferred from --finetune')
    
    args = parser.parse_args()
    
    # Determine model_type
    if args.model_type:
        model_type = args.model_type
    else:
        model_type = 'ft' if args.finetune else 'scr'
    
    # Construct file paths (same as train_t5.py)
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    
    # Also check model_output folder as fallback
    model_sql_path_alt = os.path.join(f'model_output/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path_alt = os.path.join(f'model_output/t5_{model_type}_{args.experiment_name}_dev.pkl')
    
    # Load data
    data_folder = "data"
    
    # Load NL queries
    nl_path = os.path.join(data_folder, "dev.nl")
    with open(nl_path, 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    # Load gold SQL
    gold_sql_path = os.path.join(data_folder, "dev.sql")
    gold_sql = read_queries(gold_sql_path)
    
    # Load model predictions
    print(f"Looking for model predictions...")
    print(f"  Primary: {model_sql_path}")
    print(f"  Fallback: {model_sql_path_alt}")
    
    if os.path.exists(model_sql_path):
        model_sql = read_queries(model_sql_path)
        print(f"✓ Loaded SQL from: {model_sql_path}")
    elif os.path.exists(model_sql_path_alt):
        model_sql = read_queries(model_sql_path_alt)
        model_sql_path = model_sql_path_alt
        print(f"✓ Loaded SQL from: {model_sql_path_alt}")
    else:
        print(f"✗ Model predictions not found. Tried:")
        print(f"  - {model_sql_path}")
        print(f"  - {model_sql_path_alt}")
        return
    
    # Load gold records (ground truth)
    gold_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    print(f"Looking for gold records...")
    print(f"  Path: {gold_record_path}")
    
    gold_records = None
    if os.path.exists(gold_record_path):
        with open(gold_record_path, 'rb') as f:
            gold_records, _ = pickle.load(f)
        print(f"✓ Loaded gold records from: {gold_record_path}")
    else:
        print(f"⚠ Gold records file not found. Will compare SQL strings instead.")
        print(f"  Tried: {gold_record_path}\n")
    
    # Load error messages from records file
    print(f"Looking for model records/error messages...")
    print(f"  Primary: {model_record_path}")
    print(f"  Fallback: {model_record_path_alt}")
    
    model_records = None
    error_msgs = None
    if os.path.exists(model_record_path):
        with open(model_record_path, 'rb') as f:
            model_records, error_msgs = pickle.load(f)
        print(f"✓ Loaded model records/errors from: {model_record_path}\n")
    elif os.path.exists(model_record_path_alt):
        with open(model_record_path_alt, 'rb') as f:
            model_records, error_msgs = pickle.load(f)
        model_record_path = model_record_path_alt
        print(f"✓ Loaded model records/errors from: {model_record_path_alt}\n")
    else:
        print(f"⚠ Model records file not found. Will only show SQL mismatches (not SQL errors).")
        print(f"  Tried: {model_record_path}")
        print(f"  Tried: {model_record_path_alt}\n")
    
    # Verify lengths
    assert len(nl_queries) == len(gold_sql) == len(model_sql), \
        f"Mismatch: NL={len(nl_queries)}, Gold={len(gold_sql)}, Model={len(model_sql)}"
    
    if gold_records:
        assert len(gold_records) == len(nl_queries), \
            f"Mismatch: Gold records={len(gold_records)}, NL queries={len(nl_queries)}"
    
    if model_records:
        assert len(model_records) == len(model_sql), \
            f"Mismatch: Model records={len(model_records)}, Model SQL={len(model_sql)}"
    
    if error_msgs:
        assert len(error_msgs) == len(model_sql), \
            f"Mismatch: Error messages={len(error_msgs)}, Model SQL={len(model_sql)}"
    
    # Find record mismatches (where fetched records differ)
    # Only care about cases where actual results differ, not just SQL string differences
    mismatches = []
    if gold_records and model_records:
        for i, (nl, gold_sql_str, pred_sql_str, gold_rec, pred_rec) in enumerate(
            zip(nl_queries, gold_sql, model_sql, gold_records, model_records)
        ):
            # Compare records (convert to sets of tuples for comparison)
            # Skip if there was an error executing the predicted SQL
            if error_msgs and error_msgs[i] and error_msgs[i].strip():
                continue  # Skip queries with SQL errors (handled separately)
            
            # Compare records
            gold_rec_set = set(tuple(row) for row in gold_rec) if gold_rec else set()
            pred_rec_set = set(tuple(row) for row in pred_rec) if pred_rec else set()
            
            if gold_rec_set != pred_rec_set:
                mismatches.append((i, nl, gold_sql_str, pred_sql_str, gold_rec, pred_rec))
    else:
        # Fallback: compare SQL strings if records not available
        print("⚠ Records not available. Falling back to SQL string comparison.")
        for i, (nl, gold, pred) in enumerate(zip(nl_queries, gold_sql, model_sql)):
            if gold.strip() != pred.strip():
                mismatches.append((i, nl, gold, pred, None, None))  # None for records
    
    # Find queries with SQL errors (queries that failed to execute)
    error_queries = []
    if error_msgs:
        for i, (nl, gold, pred, error_msg) in enumerate(zip(nl_queries, gold_sql, model_sql, error_msgs)):
            if error_msg and error_msg.strip():  # Non-empty error message means SQL error
                error_queries.append((i, nl, gold, pred, error_msg))
    
    print(f"Total queries: {len(nl_queries)}")
    if gold_records and model_records:
        print(f"Queries with record mismatches (different results): {len(mismatches)}")
        print(f"Record match rate: {(len(nl_queries) - len(mismatches)) / len(nl_queries) * 100:.2f}%")
    else:
        print(f"Queries with SQL mismatches: {len(mismatches)}")
        print(f"SQL match rate: {(len(nl_queries) - len(mismatches)) / len(nl_queries) * 100:.2f}%")
    if error_msgs:
        print(f"Queries with SQL errors: {len(error_queries)}")
        print(f"SQL error rate: {len(error_queries) / len(nl_queries) * 100:.2f}%")
    
    # Create output filenames based on experiment name
    mismatches_filename = f"{args.experiment_name}_mismatches.txt"
    errors_filename = f"error_{args.experiment_name}.txt"
    mismatches_path = os.path.join("mismatches", mismatches_filename)
    errors_path = os.path.join("mismatches", errors_filename)
    
    # Create mismatches directory if it doesn't exist
    os.makedirs("mismatches", exist_ok=True)
    
    # Save record mismatches to file
    print(f"\nSaving record mismatches to: {mismatches_path}")
    with open(mismatches_path, 'w') as f:
        f.write("="*80 + "\n")
        if gold_records and model_records:
            f.write(f"RECORD MISMATCHES - Experiment: {args.experiment_name}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total queries: {len(nl_queries)}\n")
            f.write(f"Queries with record mismatches (different results): {len(mismatches)}\n")
            f.write(f"Record match rate: {(len(nl_queries) - len(mismatches)) / len(nl_queries) * 100:.2f}%\n")
            f.write("\n" + "="*80 + "\n")
            f.write("ALL RECORD MISMATCHES (Different fetched results)\n")
            f.write("="*80 + "\n\n")
        else:
            f.write(f"SQL MISMATCHES - Experiment: {args.experiment_name}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total queries: {len(nl_queries)}\n")
            f.write(f"Queries with SQL mismatches: {len(mismatches)}\n")
            f.write(f"SQL match rate: {(len(nl_queries) - len(mismatches)) / len(nl_queries) * 100:.2f}%\n")
            f.write("\n" + "="*80 + "\n")
            f.write("ALL SQL MISMATCHES (Predicted SQL != Gold SQL)\n")
            f.write("="*80 + "\n\n")
        
        for idx, (i, nl, gold, pred, gold_rec, pred_rec) in enumerate(mismatches, 1):
            f.write(f"Example {idx} (Index {i}):\n")
            f.write(f"NL Query: {nl}\n")
            f.write(f"\nGold SQL:\n{gold}\n")
            f.write(f"\nPredicted SQL:\n{pred}\n")
            if gold_records and model_records:
                f.write(f"\nGold Records ({len(gold_rec) if gold_rec else 0} rows):\n")
                if gold_rec:
                    for row in gold_rec[:10]:  # Show first 10 rows
                        f.write(f"  {row}\n")
                    if len(gold_rec) > 10:
                        f.write(f"  ... ({len(gold_rec) - 10} more rows)\n")
                else:
                    f.write("  (empty)\n")
                f.write(f"\nPredicted Records ({len(pred_rec) if pred_rec else 0} rows):\n")
                if pred_rec:
                    for row in pred_rec[:10]:  # Show first 10 rows
                        f.write(f"  {row}\n")
                    if len(pred_rec) > 10:
                        f.write(f"  ... ({len(pred_rec) - 10} more rows)\n")
                else:
                    f.write("  (empty)\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    if gold_records and model_records:
        print(f"✓ Saved {len(mismatches)} record mismatches to {mismatches_path}")
    else:
        print(f"✓ Saved {len(mismatches)} SQL mismatches to {mismatches_path}")
    
    # Save SQL errors to file (if available)
    if error_msgs and error_queries:
        print(f"\nSaving SQL errors to: {errors_path}")
        with open(errors_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"SQL ERRORS - Experiment: {args.experiment_name}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total queries: {len(nl_queries)}\n")
            f.write(f"Queries with SQL errors: {len(error_queries)}\n")
            f.write(f"SQL error rate: {len(error_queries) / len(nl_queries) * 100:.2f}%\n")
            f.write("\n" + "="*80 + "\n")
            f.write("ALL QUERIES WITH SQL ERRORS (Failed to execute)\n")
            f.write("="*80 + "\n\n")
            
            for idx, (i, nl, gold, pred, error_msg) in enumerate(error_queries, 1):
                f.write(f"Example {idx} (Index {i}):\n")
                f.write(f"NL Query: {nl}\n")
                f.write(f"\nGold SQL:\n{gold}\n")
                f.write(f"\nPredicted SQL:\n{pred}\n")
                f.write(f"\nError Message:\n{error_msg}\n")
                f.write("\n" + "-"*80 + "\n\n")
        
        print(f"✓ Saved {len(error_queries)} queries with SQL errors to {errors_path}")
    
    # Print all SQL errors to console (if available)
    if error_msgs and error_queries:
        print("\n" + "="*80)
        print(f"ALL {len(error_queries)} QUERIES WITH SQL ERRORS")
        print("="*80 + "\n")
        
        for idx, (i, nl, gold, pred, error_msg) in enumerate(error_queries, 1):
            print(f"Example {idx} (Index {i}):")
            print(f"NL Query: {nl}")
            print(f"\nGold SQL:")
            print(gold)
            print(f"\nPredicted SQL:")
            print(pred)
            print(f"\nError Message:")
            print(error_msg)
            print("\n" + "-"*80 + "\n")
    
    # Print all record mismatches to console
    print("\n" + "="*80)
    if gold_records and model_records:
        print(f"ALL {len(mismatches)} RECORD MISMATCHES (Different Results)")
    else:
        print(f"ALL {len(mismatches)} SQL MISMATCHES")
    print("="*80 + "\n")
    
    for idx, (i, nl, gold, pred, gold_rec, pred_rec) in enumerate(mismatches, 1):
        print(f"Example {idx} (Index {i}):")
        print(f"NL Query: {nl}")
        print(f"\nGold SQL:")
        print(gold)
        print(f"\nPredicted SQL:")
        print(pred)
        if gold_records and model_records:
            print(f"\nGold Records ({len(gold_rec) if gold_rec else 0} rows):")
            if gold_rec:
                for row in gold_rec[:5]:  # Show first 5 rows in console
                    print(f"  {row}")
                if len(gold_rec) > 5:
                    print(f"  ... ({len(gold_rec) - 5} more rows)")
            else:
                print("  (empty)")
            print(f"\nPredicted Records ({len(pred_rec) if pred_rec else 0} rows):")
            if pred_rec:
                for row in pred_rec[:5]:  # Show first 5 rows in console
                    print(f"  {row}")
                if len(pred_rec) > 5:
                    print(f"  ... ({len(pred_rec) - 5} more rows)")
            else:
                print("  (empty)")
        print("\n" + "-"*80 + "\n")

if __name__ == '__main__':
    main()

