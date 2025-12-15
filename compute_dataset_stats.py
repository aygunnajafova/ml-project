"""
Script to compute dataset statistics for Table 1 (Before Preprocessing) and Table 2 (After Preprocessing).

Table 1: Statistics on raw data (tokenized but no prefix)
Table 2: Statistics on preprocessed data (with "translate to SQL: " prefix added to NL queries)
"""

import os
from transformers import T5TokenizerFast
from load_data import load_lines, preprocess_nl_query


def compute_stats(nl_queries, sql_queries, tokenizer, add_prefix=False):
    """
    Compute dataset statistics for NL and SQL queries.
    
    Args:
        nl_queries: List of natural language query strings
        sql_queries: List of SQL query strings
        tokenizer: T5 tokenizer instance
        add_prefix: If True, add "translate to SQL: " prefix to NL queries
    
    Returns:
        Dictionary with statistics:
        - num_examples: Number of examples
        - mean_nl_length: Mean NL sentence length (in tokens)
        - mean_sql_length: Mean SQL query length (in tokens)
        - nl_vocab_size: Vocabulary size (unique tokens in NL)
        - sql_vocab_size: Vocabulary size (unique tokens in SQL)
    """
    num_examples = len(nl_queries)
    
    # Tokenize NL queries (with or without prefix)
    nl_lengths = []
    nl_vocab = set()
    
    for nl in nl_queries:
        if add_prefix:
            nl = preprocess_nl_query(nl)
        tokens = tokenizer(nl).input_ids
        nl_lengths.append(len(tokens))
        nl_vocab.update(tokens)
    
    # Tokenize SQL queries
    sql_lengths = []
    sql_vocab = set()
    
    for sql in sql_queries:
        tokens = tokenizer(sql).input_ids
        sql_lengths.append(len(tokens))
        sql_vocab.update(tokens)
    
    # Compute statistics
    mean_nl_length = sum(nl_lengths) / len(nl_lengths) if nl_lengths else 0
    mean_sql_length = sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0
    nl_vocab_size = len(nl_vocab)
    sql_vocab_size = len(sql_vocab)
    
    return {
        'num_examples': num_examples,
        'mean_nl_length': mean_nl_length,
        'mean_sql_length': mean_sql_length,
        'nl_vocab_size': nl_vocab_size,
        'sql_vocab_size': sql_vocab_size
    }


def print_table(title, train_stats, dev_stats):
    """Print statistics in a formatted table."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'Metric':<40} {'Train':<20} {'Dev':<20}")
    print(f"{'-'*80}")
    print(f"{'Number of examples':<40} {train_stats['num_examples']:<20} {dev_stats['num_examples']:<20}")
    print(f"{'Mean sentence length (NL)':<40} {train_stats['mean_nl_length']:.2f}{'':<15} {dev_stats['mean_nl_length']:.2f}{'':<15}")
    print(f"{'Mean SQL length':<40} {train_stats['mean_sql_length']:.2f}{'':<15} {dev_stats['mean_sql_length']:.2f}{'':<15}")
    print(f"{'Vocabulary size (NL)':<40} {train_stats['nl_vocab_size']:<20} {dev_stats['nl_vocab_size']:<20}")
    print(f"{'Vocabulary size (SQL)':<40} {train_stats['sql_vocab_size']:<20} {dev_stats['sql_vocab_size']:<20}")
    print(f"{'='*80}\n")


def main():
    # Setup
    data_folder = "data"
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    
    # Load data
    print("Loading data...")
    train_nl = load_lines(os.path.join(data_folder, "train.nl"))
    train_sql = load_lines(os.path.join(data_folder, "train.sql"))
    dev_nl = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_sql = load_lines(os.path.join(data_folder, "dev.sql"))
    
    print(f"Loaded {len(train_nl)} train examples and {len(dev_nl)} dev examples")
    
    # Compute Table 1: Before Preprocessing (raw tokenization, no prefix)
    print("\nComputing Table 1: Before Preprocessing...")
    train_stats_before = compute_stats(train_nl, train_sql, tokenizer, add_prefix=False)
    dev_stats_before = compute_stats(dev_nl, dev_sql, tokenizer, add_prefix=False)
    print_table("Table 1: Before Preprocessing", train_stats_before, dev_stats_before)
    
    # Compute Table 2: After Preprocessing (with prefix on NL)
    print("Computing Table 2: After Preprocessing...")
    train_stats_after = compute_stats(train_nl, train_sql, tokenizer, add_prefix=True)
    dev_stats_after = compute_stats(dev_nl, dev_sql, tokenizer, add_prefix=True)
    print_table("Table 2: After Preprocessing", train_stats_after, dev_stats_after)


if __name__ == "__main__":
    main()

