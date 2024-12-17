import time
import os
from decimal import ROUND_DOWN, Decimal
from dydx3 import Client
from dydx3.constants import API_HOST_SEPOLIA
from dydx3.constants import MARKET_ETH_USD   # Ethereum, Bitcoin, XMR
from dydx3.constants import NETWORK_ID_SEPOLIA
from dydx3.constants import ORDER_SIDE_BUY, ORDER_SIDE_SELL
from dydx3.constants import ORDER_TYPE_LIMIT
from dydx3.constants import ORDER_TYPE_TRAILING_STOP
from dydx3.constants import ORDER_STATUS_OPEN
from dydx3.constants import POSITION_STATUS_OPEN
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers= [
                        logging.FileHandler('interface.log'),
                        logging.StreamHandler()
                    ])

# Potential to reduce steps regarding client initializing between functions

def initialize_client():
    """
    Initializes and returns a dYdX client with the environment's API credentials.

    :return: Initialized dYdX Client object.
    """
    api_key = os.getenv('testdydxKey')
    api_secret = os.getenv('testdydxSecret')
    passphrase = os.getenv('testdydxPassphrase')
    ethereum_private_key = os.getenv('testmaskKey')

    return Client(
        network_id=NETWORK_ID_SEPOLIA,
        host=API_HOST_SEPOLIA,
        eth_private_key=ethereum_private_key,
        api_key_credentials={
            'key': api_key,
            'secret': api_secret,
            'passphrase': passphrase,
        },
    )

def place_limit_order(side_input, size, price, post_only=False, limit_fee='0.0015', expiration_seconds=300, position_id=None):
    logging.info(f"Placing limit order: side={side_input}, size={size}, price={price}")
    
    client = initialize_client()
    stark_key_data = client.onboarding.derive_stark_key()
    client.stark_private_key = stark_key_data['private_key']
    
    if not position_id:
        account_response = client.private.get_account().data
        position_id = account_response['account']['positionId']

    if side_input.lower() == 'buy':
        side = ORDER_SIDE_BUY
    elif side_input.lower() == 'sell':
        side = ORDER_SIDE_SELL
    else:
        raise ValueError("Invalid order side. Must be 'buy' or 'sell'.")

    tickSize_value = 0.001

    tickSize = Decimal(str(tickSize_value))  # The tick size, also as Decimal for precision

    # Convert indexPrice to Decimal for precise arithmetic
    size_decimal = Decimal(str(size))

    # Round down to nearest tickSize to ensure divisibility and return
    new_size_rounded = (size_decimal / tickSize).quantize(Decimal('1.'), rounding=ROUND_DOWN) * tickSize

    expiration_time = int(time.time()) + expiration_seconds


    logging.info(f"New size rounded: {new_size_rounded}, from {size} to {size_decimal}")
    order_params = {
        'position_id': position_id,
        'market': MARKET_ETH_USD,
        'side': side,
        'order_type': ORDER_TYPE_LIMIT,
        'size': str(new_size_rounded),
        'post_only': post_only,
        'price': str(price),
        'limit_fee': limit_fee,
        'expiration_epoch_seconds': expiration_time,
    }

    return client.private.create_order(**order_params).data

def fetch_orders():
    client = initialize_client()

    orders_response = client.private.get_orders()
    orders = orders_response.data.get('orders', [])

    return orders

def fetch_order_by_id(order_id):
    client = initialize_client()

    # orders_response = client.private.get_order_by_client_id(order_id)

    orders_response = client.private.get_order_by_id(order_id)
    order = orders_response.data.get('orders', [])

    return order

def fetch_open_orders():
    client = initialize_client()

    open_orders_response = client.private.get_orders(
        status = ORDER_STATUS_OPEN
    )

    open_orders = open_orders_response.data.get('orders', [])

    return open_orders

def fetch_positions():
    client = initialize_client()
    
    positions_response = client.private.get_positions()
    positions = positions_response.data.get('positions', [])

    return positions

def fetch_open_positions():
    client = initialize_client()
    
    open_positions_response = client.private.get_positions(
        status = POSITION_STATUS_OPEN
    )

    open_positions = open_positions_response.data.get('positions', [])

    return open_positions

def fetch_account_balance():
    client = initialize_client()
    account_response = client.private.get_account()
    balance_info = account_response.data.get('account', {})

    return {
        'equity': balance_info.get('equity'),
        'freeCollateral': balance_info.get('freeCollateral'),
        'pendingDeposits': balance_info.get('pendingDeposits'),
        'pendingWithdrawals': balance_info.get('pendingWithdrawals'),
        'quoteBalance': balance_info.get('quoteBalance')
    }

def fetch_equity():
    client = initialize_client()
    account_response = client.private.get_account()
    balance_info = account_response.data.get('account', {})

    return float(balance_info.get('equity'))

def fetch_position_size():
    client = initialize_client()
    open_positions_response = client.private.get_positions(
        status = POSITION_STATUS_OPEN
    )
    
    open_positions_response = open_positions_response.data.get('positions', [])
    
    return open_positions_response[0]['size']

def fetch_leverage():
    client = initialize_client()
    open_positions_response = client.private.get_positions(
        status = POSITION_STATUS_OPEN
    )
    balances = fetch_account_balance()
    
    open_positions_response = open_positions_response.data.get('positions', [])
    
    size = open_positions_response[0]['size']
    entry_price = open_positions_response[0]['entryPrice']
    
    position_value = abs(float(size)) * float(entry_price)
    
    effective_leverage = position_value / float(balances['equity'])
    
    return effective_leverage

def fetch_eth_market_data():
    """
    Fetches market data for Ethereum from the dYdX exchange.

    :return: A dictionary with specific market details for Ethereum.
    """
    client = initialize_client()

    # Fetch all market data
    market_data_response = client.public.get_markets()
    market_data = market_data_response.data.get('markets', {})

    # Extract ETH-USD market data
    eth_data = market_data.get('ETH-USD', {})

    # Extract specific details
    eth_details = {
        'indexPrice': eth_data.get('indexPrice'),
        'oraclePrice': eth_data.get('oraclePrice'),
        'priceChange24H': eth_data.get('priceChange24H'),
        'openInterest': eth_data.get('openInterest'),
        'volume24H': eth_data.get('volume24H'),
        'trades24H': eth_data.get('trades24H'),
        'nextFundingRate': eth_data.get('nextFundingRate'),
        'nextFundingAt': eth_data.get('nextFundingAt')
    }

    return eth_details

def fetch_eth_price():
    """
    Fetches market data for Ethereum from the dYdX exchange.

    :return: A dictionary with specific market details for Ethereum.
    """
    client = initialize_client()

    # Fetch all market data
    market_data_response = client.public.get_markets()
    market_data = market_data_response.data.get('markets', {})

    # Extract ETH-USD market data
    eth_data = market_data.get('ETH-USD', {})

    # Extract specific details
    eth_details = {
        'indexPrice': eth_data.get('indexPrice'),
    }

    index_price = float(eth_details['indexPrice'])
    
    return float(index_price)

def cancel_order(order_id):
    """
    Cancels an order on the dYdX exchange given a position_id.

    :param position_id: The ID of the position associated with the order to cancel.
    :return: The response from the cancel order request.
    """
    client = initialize_client()

    # Cancel the order
    cancel_response = client.private.cancel_order(order_id)
    
    return cancel_response.data

def cancel_all_orders():
    """
    Cancels all open orders on the dYdX exchange.

    :return: The response from the cancel all orders request.
    """
    client = initialize_client()

    # Cancel all orders
    cancel_response = client.private.cancel_all_orders()
    
    return cancel_response.data

def place_trailing_stop_order(size, side_input, price, trailing_percent=0.005, market=MARKET_ETH_USD, limit_fee='0.0015', time_in_force='GTT', expiration_seconds=3600, position_id=None):
    """
    NOTES: I think this order should be generated with the same info at the same time as a limit order. If it is a sell, the trailing percent has to be negative.
    Time In Force Acronyms:
    Good Till Cancelled (GTC): The order remains active until it is either filled or manually cancelled by the trader.
    Good Till Time (GTT): The order will expire at a specific time unless it is filled or cancelled before that time.
    Immediate or Cancel (IOC): The order must be executed immediately. Any portion of the order that cannot be filled immediately is cancelled.
    Fill or Kill (FOK): The order must be filled in its entirety immediately, or it is entirely cancelled.

    Place a trailing stop order on dYdX.

    :param market: Market symbol, e.g., 'ETH-USD'.
    :param size: Amount of the asset to sell.
    :param trailing_percent: The percentage the trigger price should trail the market price.
    :param side: 'SELL' for a stop loss order (default). Use 'BUY' for a stop gain order.
    :param time_in_force: Order's time in force, 'GTT' (Good 'Til Time) by default.
    :param expiration_seconds: Number of seconds until the order expires, default is 3600 (1 hour).
    :return: The response from the order request.
    """
    client = initialize_client()
    stark_key_data = client.onboarding.derive_stark_key()
    client.stark_private_key = stark_key_data['private_key']

    if not position_id:
        account_response = client.private.get_account().data
        position_id = account_response['account']['positionId']

    if side_input.lower() == 'buy':
        side = ORDER_SIDE_BUY
    elif side_input.lower() == 'sell':
        side = ORDER_SIDE_SELL
    else:
        raise ValueError("Invalid order side. Must be 'buy' or 'sell'.")
    
    order_params = {
        'position_id': position_id,
        'market': market,
        'side': side,
        'order_type': ORDER_TYPE_TRAILING_STOP,
        'size': str(size),
        'price': str(price),
        'trailing_percent': str(trailing_percent),
        'limit_fee': limit_fee,
        'post_only': False,
        'time_in_force': time_in_force,
        'expiration_epoch_seconds': int(time.time()) + expiration_seconds,
    }

    order_response = client.private.create_order(**order_params)
    return order_response.data

def calculate_new_price(indexPrice, operation='subtract', buffer_value = 2, tickSize_value = 0.1):
    """
    Calculates a new price by either subtracting or adding a buffer to the indexPrice,
    ensuring the result is rounded down to the nearest tick size.

    :param indexPrice: The original price as a numeric value or string for precision.
    :param buffer_value: The buffer value to adjust the price by to improve fill chance.
    :param tickSize_value: The tick size to which the new price must be rounded.
    :param operation: 'subtract' to subtract the buffer, 'add' to add the buffer.
    :return: The new price, adjusted and rounded as specified.
    """
    buffer = Decimal(str(buffer_value))  # Use Decimal for the buffer for precise arithmetic
    tickSize = Decimal(str(tickSize_value))  # The tick size, also as Decimal for precision

    # Convert indexPrice to Decimal for precise arithmetic
    indexPrice_decimal = Decimal(str(indexPrice))

    # Calculate the new price with buffer, depending on the operation
    if operation == 'subtract':
        new_price = indexPrice_decimal - buffer
    elif operation == 'add':
        new_price = indexPrice_decimal + buffer
    else:
        raise ValueError("Invalid operation. Use 'add' or 'subtract'.")

    # Round down to nearest tickSize to ensure divisibility and return
    new_price_rounded = (new_price / tickSize).quantize(Decimal('1.'), rounding=ROUND_DOWN) * tickSize
    logging.info(f"Price Rounded and buffered from {indexPrice_decimal} to {new_price_rounded}" )
    return new_price_rounded

def clear_existing_orders_and_positions():
    position_is_open = False

    # Cancel all existing orders
    cancel_orders = cancel_all_orders()
    print("Cancel Orders:", cancel_orders)

    # Delay to keep order from being placed and cancelled immediately
    time.sleep(3)

    # Fetch open positions
    open_positions = fetch_open_positions()

    # Check for open positions and extract details
    for position in open_positions:
        if position['status'] == 'OPEN':
            position_is_open = True
            market = position['market']
            side = position['side']
            size = position['size']
            # print(f"Market: {market}, Side: {side}, Size: {size}")
            break  # Exit the loop once the first open position is found

    # print('Boolean', position_is_open)

    # If there is an open position, prepare to close it
    if position_is_open:
        if side == 'LONG':
            side = 'sell'
            operation = 'subtract'
        elif side == 'SHORT':
            side = 'buy'
            operation = 'add'

        # print('Side:', side)

        # Fetch market data
        market_data = fetch_eth_market_data()
        # print('Market Data:', market_data)

        # Extract and convert index price to a float
        index_price = float(market_data['indexPrice'])
        # print('Index Price:', index_price)

        # Calculate the new price
        price = calculate_new_price(index_price, operation)
        # print('Calculated Price:', price)

        # Place a limit order to close the position
        close_position_order = place_limit_order(side, size, price)
        print('Order Return:', close_position_order)

    # Fetch open orders
    open_orders = fetch_open_orders()
    open_positions = fetch_open_positions()

    # print("Open Orders:", open_orders)
    # print("Positions:", open_positions)

    return open_orders, open_positions

# Possible Withdrawal Function
# from dydx3.constants import ASSET_USDC

# withdrawal = client.private.create_withdrawal(
#   position_id=1, # required for creating the withdrawal signature
#   amount='100',
#   asset=ASSET_USDC,
#   expiration_epoch_seconds=1613988637,
# )


# if __name__ == "__main__":
#     # Test place_limit_order function
#     print("Placing a limit order...")
#     try:
#         order_result = place_limit_order(
#             side_input='buy',  # or 'sell'
#             size='0.5',  # Specify the size
#             price='3407.1',  # Specify the price
#             post_only=False,
#             limit_fee='0.0015',
#             expiration_seconds=300  # 5 minutes
#         )
#         print("Limit Order Result:")
#         print(order_result)
#     except Exception as e:
#         print(f"Error placing limit order: {e}")

#     # Test fetch_current_orders_and_positions function
#     print("\nFetching current orders and positions...")
#     try:
#         orders, positions = fetch_current_orders_and_positions()
#         print("Current Orders:")
#         print(orders)
#         print("Current Positions:")
#         print(positions)
#     except Exception as e:
#         print(f"Error fetching orders and positions: {e}")

#     # Test fetch_account_balance function
#     print("\nFetching account balance...")
#     try:
#         balance_info = fetch_account_balance()
#         print("Account Balance:")
#         print(balance_info)
#     except Exception as e:
#         print(f"Error fetching account balance: {e}")

#     print("\nFetching Ethereum market data...")
#     try:
#         eth_market_data = fetch_eth_market_data()
#         print("Ethereum Market Data:")
#         print(eth_market_data)
#     except Exception as e:
#         print(f"Error fetching Ethereum market data: {e}")

# if __name__ == "__main__":
#     # Other test functions...

#     # Test cancel_order function
#     print("\nCancelling an order...")
#     try:
#         position_id_to_cancel = '26f8721eb69f7f885dd6f3a7e80a395ffbb8dc0d6ce801376a7d7deb71401f5'  # Replace with the actual position ID you want to cancel
#         cancel_result = cancel_order(position_id_to_cancel)
#         print("Cancel Order Result:")
#         print(cancel_result)
#     except Exception as e:
#         print(f"Error cancelling order: {e}")

# if __name__ == "__main__":
#     # Test cancel_all_orders function
#     print("\nCancelling all orders...")
#     try:
#         cancel_all_result = cancel_all_orders()
#         print("Cancel All Orders Result:")
#         print(cancel_all_result)
#     except Exception as e:
#         print(f"Error cancelling all orders: {e}")

# if __name__ == "__main__":
#     print("\nPlacing a trailing stop order...")
#     try:
#         trailing_stop_result = place_trailing_stop_order(
#             price='3800',
#             size='0.05',  # Specify the size
#             side_input='buy',  # 'buy' or 'sell'
#             trailing_percent=0.005,  # Specify the trailing percent
#             market='ETH-USD',  # Specify the market
#             time_in_force='GTT',  # Specify time in force
#             expiration_seconds=7200  # 2 hours
#         )
#         print("Trailing Stop Order Result:")
#         print(trailing_stop_result)
#     except Exception as e:
#         print(f"Error placing trailing stop order: {e}")

# if __name__ == "__main__":
#     try:
#         # Place a limit buy order
#         print("Placing a limit buy order...")
#         limit_order_result = place_limit_order(
#             side_input='buy',  # or 'sell'
#             size='0.1',  # Specify the size
#             price='3300',  # Specify the price for limit order
#             post_only=False,
#             limit_fee='0.0015',
#             expiration_seconds=7200  # 2 hours
#         )
#         print("Limit Order Result:")
#         print(limit_order_result)

#         # Place a trailing stop buy order
#         print("\nPlacing a trailing stop buy order...")
#         trailing_stop_order_result = place_trailing_stop_order(
#             size='0.1',  # Specify the size, should match or be less than the limit order size
#             side_input='buy',  # 'buy' or 'sell'
#             price='3600',  # Specify the trigger price for trailing stop order
#             trailing_percent=0.01,  # Specify the trailing percent
#             market=MARKET_ETH_USD,  # Specify the market
#             time_in_force='GTT',  # Specify time in force
#             expiration_seconds=7200  # 2 hours
#         )
#         print("Trailing Stop Order Result:")
#         print(trailing_stop_order_result)

#     except Exception as e:
#         print(f"Error in placing orders: {e}")



