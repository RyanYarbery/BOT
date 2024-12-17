
import os
import logging
import datetime
import pandas as pd
from decimal import Decimal
from decimal import ROUND_DOWN
from dateutil.relativedelta import relativedelta
from DYDXInterface.dydx_interface import DydxInterface

dydx = DydxInterface()

# balance = dydx.fetch_account_balance()

# print(balance)

position = dydx.fetch_open_positions()
print(position)

# Load Model

# Should Ideally, cancel open orders and place limit orders to close open positions.

# balances = fetch_account_balance()
# print(balances)
# print(balances['quoteBalance'])

# open_positions = fetch_open_positions()
# print(open_positions)

# print(fetch_leverage())

# equity = fetch_equity()
# print(equity)

# clean_slate = clear_existing_orders_and_positions() 
# print('Open Orders and Postions:', clean_slate)


# tickSize_value = 0.1
# size = 6.341012063078812
# tickSize = Decimal(str(tickSize_value))  # The tick size, also as Decimal for precision
# # Convert indexPrice to Decimal for precise arithmetic
# size_decimal = Decimal(str(size))
# # Round down to nearest tickSize to ensure divisibility and return
# new_size_rounded = (size_decimal / tickSize).quantize(Decimal('1.'), rounding=ROUND_DOWN) * tickSize
# print(new_size_rounded)

# order_id = '45143018341374088'
# order = fetch_order_by_id(order_id)
# print(order)

# orders = fetch_orders()
# print(orders)

