import pandas as pd
import numpy as np
import re

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train = pd.read_csv(os.path.join(base_dir, 'dataset', 'train.csv'))
test = pd.read_csv(os.path.join(base_dir, 'dataset', 'test.csv'))

def extract_field(text, field):
    m = re.search(rf'^{field}:\s*(.*)$', str(text), re.MULTILINE)
    return m.group(1).strip() if m else ''

print("=== 25 SAMPLE ITEMS AND THEIR PRICES ===")
for i in range(25):
    name = extract_field(train.loc[i, 'catalog_content'], 'Item Name')
    val = extract_field(train.loc[i, 'catalog_content'], 'Value')
    unit = extract_field(train.loc[i, 'catalog_content'], 'Unit')
    price = train.loc[i, 'price']
    print(f"[{i:02d}] ${price:6.2f} | Val: {val:6s} | Unit: {unit:10s} | {name[:80]}")
