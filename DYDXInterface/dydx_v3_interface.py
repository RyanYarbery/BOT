import os
import time
import logging
from dydx3 import Client
from decimal import Decimal
from decimal import ROUND_DOWN
from dydx3.constants import MARKET_ETH_USD  
from dydx3.constants import API_HOST_SEPOLIA
from dydx3.constants import API_HOST_MAINNET
from dydx3.constants import ORDER_TYPE_LIMIT
from dydx3.constants import ORDER_STATUS_OPEN
from dydx3.constants import NETWORK_ID_SEPOLIA
from dydx3.constants import NETWORK_ID_MAINNET
from dydx3.constants import POSITION_STATUS_OPEN
from dydx3.constants import ORDER_TYPE_TRAILING_STOP
from dydx3.constants import ORDER_SIDE_BUY, ORDER_SIDE_SELL

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers= [
                        logging.FileHandler('interface.log'),
                        logging.StreamHandler()
                    ])

# For Testnet dydx = DydxInterface()
# For Mainnet dydx = DydxInterface(environment='main')

class DydxInterface:

    def __init__(self, environment='test'):
        """
        Initializes the dYdX client with the environment's API credentials.
        
        :param environment: Set to 'main' for mainnet, otherwise defaults to testnet.
        """
        self.environment = environment.lower()
        
        if self.environment == 'main':
            self.network_id = NETWORK_ID_MAINNET
            self.host = API_HOST_MAINNET
            self.api_key = os.getenv('dydxKey')
            self.api_secret = os.getenv('dydxSecret')
            self.passphrase = os.getenv('dydxPassphrase')
            self.ethereum_private_key = os.getenv('maskKey')
        else:
            self.network_id = NETWORK_ID_SEPOLIA
            self.host = API_HOST_SEPOLIA
            self.api_key = os.getenv('testdydxKey')
            self.api_secret = os.getenv('testdydxSecret')
            self.passphrase = os.getenv('testdydxPassphrase')
            self.ethereum_private_key = os.getenv('testmaskKey')

        self.client = Client(
            network_id=self.network_id,
            host=self.host,
            eth_private_key=self.ethereum_private_key,
            api_key_credentials={
                'key': self.api_key,
                'secret': self.api_secret,
                'passphrase': self.passphrase,
            },
        )
        self._set_stark_private_key()

    def _set_stark_private_key(self):
        stark_key_data = self.client.onboarding.derive_stark_key()
        self.client.stark_private_key = stark_key_data['private_key']

    def place_limit_order(self, side_input, size, price, post_only=False, limit_fee='0.0015', expiration_seconds=300, position_id=None):
        logging.info(f"Placing limit order: side={side_input}, size={size}, price={price}")

        if not position_id:
            account_response = self.client.private.get_account().data
            position_id = account_response['account']['positionId']

        if side_input.lower() == 'buy':
            side = ORDER_SIDE_BUY
        elif side_input.lower() == 'sell':
            side = ORDER_SIDE_SELL
        else:
            raise ValueError("Invalid order side. Must be 'buy' or 'sell'.")

        tickSize_value = 0.001
        tickSize = Decimal(str(tickSize_value))
        size_decimal = abs(Decimal(size))
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

        order_data = self.client.private.create_order(**order_params).data
        if not order_data:
            # print("No data returned from placing limit order.")
            logging.info("No data returned from placing limit order")
        return order_data

    def fetch_orders(self):
        logging.info("Fetching Orders")
        orders_response = self.client.private.get_orders()
        orders = orders_response.data.get('orders', [])
        if not orders:
            # print("No orders to fetch.")
            logging.info("No orders to fetch")
        return orders

    def fetch_order_by_id(self, order_id):
        logging.info("Fetching order by id")
        orders_response = self.client.private.get_order_by_id(order_id)
        orders = orders_response.data.get('orders', [])
        if not orders:
            # print(f"No order found with ID {order_id}.")
            logging.info(f"No order found with ID {order_id}.")
        return orders

    def fetch_open_orders(self):
        logging.info("Fetching open orders")
        open_orders_response = self.client.private.get_orders(status=ORDER_STATUS_OPEN)
        open_orders = open_orders_response.data.get('orders', [])
        if not open_orders:
            # print("No open orders to fetch.")
            logging.info("No open orders to fetch")
        return open_orders

    def fetch_positions(self):
        logging.info("Fetching positions")
        positions_response = self.client.private.get_positions()
        positions = positions_response.data.get('positions', [])
        if not positions:
            # print("No positions to fetch.")
            logging.info("No positions to fetch")
        return positions

    def fetch_open_positions(self):
        logging.info("Fetching open positions")
        open_positions_response = self.client.private.get_positions(status=POSITION_STATUS_OPEN)
        open_positions = open_positions_response.data.get('positions', [])
        if not open_positions:
            # print("No open positions to fetch.")
            logging.info("No open positions to fetch")
        return open_positions

    def fetch_account_balance(self):
        logging.info("Fetching account balance info")
        account_response = self.client.private.get_account()
        balance_info = account_response.data.get('account', {})
        if not balance_info:
            # print("No account balance information available.")
            logging.info("No account balance info available")
        return {
            'equity': balance_info.get('equity'),
            'freeCollateral': balance_info.get('freeCollateral'),
            'pendingDeposits': balance_info.get('pendingDeposits'),
            'pendingWithdrawals': balance_info.get('pendingWithdrawals'),
            'quoteBalance': balance_info.get('quoteBalance')
        }

    def fetch_equity(self):
        logging.info("Fetching equity")
        account_response = self.client.private.get_account()
        balance_info = account_response.data.get('account', {})
        equity = float(balance_info.get('equity', 0))
        if equity == 0:
            # print("Failed to fetch equity")
            logging.info("Failed to fetch equity")
        print('Equity:', equity)
        return equity

    def fetch_position_size(self):
        logging.info("Fetching position size")
        open_positions = self.fetch_open_positions()
        size = open_positions[0]['size'] if open_positions else None
        size = abs(Decimal(size))
        if not size:
            # print("No open positions found to fetch size.")
            logging.info("No positions found to fetch size")
        return size

    def fetch_leverage(self):
        logging.info("Fetching leverage")
        open_positions = self.fetch_open_positions()
        if not open_positions:
            # print("No open positions found to calculate leverage.")
            logging.info("No open positions found to calculate leverage")
            return None

        balances = self.fetch_account_balance()
        size = open_positions[0]['size']
        entry_price = open_positions[0]['entryPrice']
        position_value = abs(float(size)) * float(entry_price)
        leverage = position_value / float(balances['equity'])
        return leverage

    def fetch_eth_market_data(self):
        logging.info("Fetching market data")
        market_data_response = self.client.public.get_markets()
        market_data = market_data_response.data.get('markets', {})
        eth_data = market_data.get('ETH-USD', {})
        if not eth_data:
            # print("No Ethereum market data found.")
            logging.info("No ethereum market data found")
        return eth_data

    def fetch_eth_price(self):
        logging.info("Fetching ethereum price")
        eth_data = self.fetch_eth_market_data()
        price = float(eth_data.get('indexPrice', 0))
        if price == 0:
            # print("Fetching ethereum price failed.")
            logging.info("Fetching ethereum price failed or price was 0 which is maybe more strange")
        return price

    def cancel_order(self, order_id):
        logging.info(f"Cancelling order with id:{order_id}")
        cancel_response = self.client.private.cancel_order(order_id)
        cancel_data = cancel_response.data
        if not cancel_data:
            # print(f"No data returned for order cancellation with ID {order_id}.")
            logging.info(f"No data returned for order cancellation with ID {order_id}")
        return cancel_data

    def cancel_all_orders(self):
        logging.info("Cancelling all orders")
        cancel_response = self.client.private.cancel_all_orders()
        cancel_data = cancel_response.data
        if not cancel_data:
            # print("No orders to cancel.")
            logging.info(f"No orders to cancel")
        return cancel_data

    def place_trailing_stop_order(self, size, side_input, price, trailing_percent=0.005, market=MARKET_ETH_USD, limit_fee='0.0015', time_in_force='GTT', expiration_seconds=3600, position_id=None):
        logging.info(f"Placing trailing stop order of Size:{size}, Type:{side_input}, Price:{price}")
        if not position_id:
            account_response = self.client.private.get_account().data
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

        order_data = self.client.private.create_order(**order_params).data
        if not order_data:
            # print("No data returned from placing trailing stop order.")
            logging.info("No data return from placing trailing stop order.")
        return order_data

    def calculate_new_price(self, indexPrice, operation='subtract', buffer_value=2, tickSize_value=0.1):
        logging.info(f"Calculating new price from {indexPrice} buffered by {operation}{buffer_value} and rounding to the {tickSize_value}")
        buffer = Decimal(str(buffer_value))
        tickSize = Decimal(str(tickSize_value))
        indexPrice_decimal = Decimal(str(indexPrice))

        if operation == 'subtract':
            new_price = indexPrice_decimal - buffer
        elif operation == 'add':
            new_price = indexPrice_decimal + buffer
        else:
            raise ValueError("Invalid operation. Use 'add' or 'subtract'.")

        new_price_rounded = (new_price / tickSize).quantize(Decimal('1.'), rounding=ROUND_DOWN) * tickSize
        logging.info(f"Price Rounded and buffered from {indexPrice_decimal} to {new_price_rounded}")
        return new_price_rounded

    def clear_existing_orders_and_positions(self):
        logging.info("Clearing existing orders and positions")
        position_is_open = False

        cancel_orders = self.cancel_all_orders()
        logging.info("Cancel Orders: %s", cancel_orders)

        time.sleep(3)
        open_positions = self.fetch_open_positions()

        for position in open_positions:
            if position['status'] == 'OPEN':
                position_is_open = True
                market = position['market']
                side = position['side']
                size = abs(Decimal(position['size']))
                break

        if position_is_open:
            if side == 'LONG':
                side = 'sell'
                operation = 'subtract'
            elif side == 'SHORT':
                side = 'buy'
                operation = 'add'

            market_data = self.fetch_eth_market_data()
            index_price = float(market_data['indexPrice'])
            price = self.calculate_new_price(index_price, operation)
            close_position_order = self.place_limit_order(side, size, price)
            logging.info("Order Return: %s", close_position_order)

        open_orders = self.fetch_open_orders()
        open_positions = self.fetch_open_positions()

        return open_orders, open_positions