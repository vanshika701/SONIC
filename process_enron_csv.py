import os
import csv
import re
from collections import defaultdict
import networkx as nx

def build_enron_edgelist(csv_path, out_path):
    print(f"[Enron] Parsing raw CSV to create edgelist. This may take a couple of minutes...")
    
    from_re = re.compile(r'^From:\s*(.+)$', re.MULTILINE)
    to_re = re.compile(r'^To:\s*(.+)$', re.MULTILINE)
    
    edges = set()
    
    # Need to handle large CSV fields
    csv.field_size_limit(2147483647)
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip header
        except StopIteration:
            pass
            
        count = 0
        for row in reader:
            if len(row) < 2: continue
            msg = row[1]
            
            from_match = from_re.search(msg)
            to_match = to_re.search(msg)
            
            if from_match and to_match:
                sender = from_match.group(1).strip()
                receivers = [r.strip() for r in to_match.group(1).split(',')]
                
                # filter to mostly keep enron addresses or clean data if needed
                # for now, keep all non-empty
                for r in receivers:
                    if r and sender != r:  # avoid self loops trivially
                        # Basic cleanup
                        r = r.replace('\r', '').replace('\n', '').replace('\t', '')
                        sender = sender.replace('\r', '').replace('\n', '').replace('\t', '')
                        edges.add((sender, r))
                        
            count += 1
            if count % 100000 == 0:
                print(f"[Enron] Processed {count} emails...")

    print(f"[Enron] Extracted {len(edges)} unique directed edges.")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\n")
    
    print(f"[Enron] Edgelist saved to {out_path}")

if __name__ == "__main__":
    csv_path = "data/raw/emails.csv"
    out_path = "data/raw/enron_edgelist.txt"
    if os.path.exists(csv_path):
        build_enron_edgelist(csv_path, out_path)
    else:
        print(f"Error: {csv_path} not found.")
