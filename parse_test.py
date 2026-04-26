import csv
import re
from collections import defaultdict

infile = 'data/raw/emails.csv'
outfile = 'data/raw/enron_edgelist.txt'

from_re = re.compile(r'^From:\s*(.+)$', re.MULTILINE)
to_re = re.compile(r'^To:\s*(.+)$', re.MULTILINE)

edges = set()

with open(infile, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    count = 0
    for row in reader:
        if len(row) < 2: continue
        msg = row[1]
        
        from_match = from_re.search(msg)
        to_match = to_re.search(msg)
        
        if from_match and to_match:
            sender = from_match.group(1).strip()
            receivers = [r.strip() for r in to_match.group(1).split(',')]
            for r in receivers:
                if r:
                    edges.add((sender, r))
        
        count += 1
        if count >= 10000:
            break

print(f"Extracted {len(edges)} edges from first 10k messages.")
