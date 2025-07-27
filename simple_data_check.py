import pandas as pd
import numpy as np

def analyze_data():
    print("JIGSAW REDDIT DATASET ANALYSIS")
    print("=" * 50)
    
    # Load data
    train_df = pd.read_csv('Data/train.csv')
    test_df = pd.read_csv('Data/test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train columns: {list(train_df.columns)}")
    
    # Critical metrics for few-shot learning
    print("\nCRITICAL METRICS:")
    print(f"Total training samples: {len(train_df):,}")
    print(f"Unique rules: {train_df['rule'].nunique()}")
    print(f"Unique subreddits: {train_df['subreddit'].nunique()}")
    
    # Samples per rule
    print(f"\nSAMPLES PER RULE:")
    samples_per_rule = train_df['rule'].value_counts()
    for rule, count in samples_per_rule.items():
        print(f"  {rule}: {count:,} samples")
    
    # Class balance
    violation_rate = train_df['rule_violation'].mean()
    print(f"\nCLASS BALANCE:")
    print(f"Violation rate: {violation_rate:.3f} ({violation_rate*100:.1f}%)")
    
    # Rule-specific violation rates
    print(f"\nVIOLATION RATES BY RULE:")
    for rule in train_df['rule'].unique():
        rule_data = train_df[train_df['rule'] == rule]
        rate = rule_data['rule_violation'].mean()
        count = len(rule_data)
        print(f"  {rule}: {rate:.3f} ({rate*100:.1f}%) - {count} samples")
    
    # Assessment
    min_samples = samples_per_rule.min()
    total_samples = len(train_df)
    
    print(f"\nASSESSMENT:")
    if total_samples < 1000:
        print("VERY LIMITED DATA - Pure few-shot learning required")
    elif min_samples < 100:
        print("IMBALANCED - Some rules need few-shot approaches")
    else:
        print("SUFFICIENT - Traditional ML possible")
    
    print(f"Min samples per rule: {min_samples}")
    print(f"Max samples per rule: {samples_per_rule.max()}")
    
    # Show first few examples
    print(f"\nFIRST FEW TRAINING EXAMPLES:")
    print(train_df[['rule', 'subreddit', 'rule_violation', 'body']].head())
    
    return {
        'total_samples': total_samples,
        'unique_rules': train_df['rule'].nunique(),
        'samples_per_rule': samples_per_rule.to_dict(),
        'violation_rate': violation_rate
    }

if __name__ == "__main__":
    results = analyze_data()
