import asyncio
from dydx_v4_interface import DydxInterface

async def main():
    
    dydx_interface = DydxInterface(environment='test')
    print('Open orders: ' + await dydx_interface.get_open_orders())  # Fetch open orders
    
if __name__ == '__main__':
    asyncio.run(main())
