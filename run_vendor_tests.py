from src.analytics import VendorAnalysis
import pandas as pd

# test 1: normal case
df = pd.DataFrame({'vendor_name': ['A', 'B', 'A'], 'contract_value': [100, 200, 100]})
print('HHI:', VendorAnalysis.herfindahl_index(df))
print('CR4:', VendorAnalysis.vendor_concentration_ratio(df, top_n=2))
print('Diversity:', VendorAnalysis.vendor_diversity_index(df))

# test 2: missing contract_value column but numeric 'amount' exists
df2 = pd.DataFrame({'vendor_name': ['X', 'Y', 'X'], 'amount': [50, 150, 100]})
print('HHI (amount):', VendorAnalysis.herfindahl_index(df2))
print('CR4 (amount):', VendorAnalysis.vendor_concentration_ratio(df2, top_n=1))
print('Diversity (amount):', VendorAnalysis.vendor_diversity_index(df2))

# test 3: total value zero
df3 = pd.DataFrame({'vendor_name': ['A', 'B'], 'contract_value': [0, 0]})
print('HHI zero total:', VendorAnalysis.herfindahl_index(df3))
print('CR4 zero total:', VendorAnalysis.vendor_concentration_ratio(df3))
print('Diversity zero total:', VendorAnalysis.vendor_diversity_index(df3))
