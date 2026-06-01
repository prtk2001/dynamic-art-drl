"""
Dynamic ART-DRL — Upstox MCX Futures Execution Layer
===================================================
Connects to the Upstox API v2 to execute live/paper trading orders for
MCX commodity futures (Gold and Crude Oil). Translates reinforcement learning
actions into actual orders and manages positions.
"""

import requests
import upstox_client
from upstox_client.rest import ApiException
from loguru import logger
from typing import Dict, Any, Optional, List
import datetime

from config.settings import (
    UPSTOX_API_KEY,
    UPSTOX_API_SECRET,
    UPSTOX_ACCESS_TOKEN
)
from config.assets import AssetConfig


class UpstoxExecutor:
    """Handles order placement, position query, and contract resolution via Upstox API v2."""
    
    def __init__(self) -> None:
        self.access_token = UPSTOX_ACCESS_TOKEN
        
        # Setup Upstox Python SDK configuration
        self.configuration = upstox_client.Configuration()
        self.configuration.access_token = self.access_token
        self.api_client = upstox_client.ApiClient(self.configuration)
        
        # Initialize sub-API instances
        self.order_api = upstox_client.OrderApi(self.api_client)
        self.portfolio_api = upstox_client.PortfolioApi(self.api_client)
        self.instruments_api = upstox_client.InstrumentsApi(self.api_client)
        
        # Cache for active instrument keys to avoid redundant lookups
        self._key_cache: Dict[str, str] = {}
        
        logger.info("UpstoxExecutor initialised.")
        
    def resolve_active_future_contract(self, symbol: str) -> Optional[str]:
        """Query Upstox search API to find the active near-month futures contract for MCX.
        
        Parameters
        ----------
        symbol : str
            Base symbol (e.g., 'GOLD', 'CRUDEOIL').
            
        Returns
        -------
        instrument_key : str or None
            Key in format 'MCX_FO|{token}' or None if not found.
        """
        # Return cached key if available
        if symbol in self._key_cache:
            return self._key_cache[symbol]
            
        try:
            # Call instrument search on MCX exchange
            response = self.instruments_api.get_instrument_search(
                query=symbol,
                exchange="MCX"
            )
            
            if not response.data:
                logger.warning("No instruments found for query '{}'", symbol)
                return None
                
            # Filter for futures contracts (usually contains 'FUT' and has an expiry date)
            # Find the contract with the closest expiry date (near-month contract)
            futures = []
            for inst in response.data:
                # Upstox symbol matches: e.g. "GOLD 26JUNFUT" or similar
                trading_symbol = inst.trading_symbol
                
                # We want FUT contract, not option (CE/PE)
                if "FUT" in trading_symbol:
                    # Get expiry date if available
                    expiry = getattr(inst, 'expiry', None)
                    futures.append((inst.instrument_key, trading_symbol, expiry))
            
            if not futures:
                logger.warning("No futures contracts found for query '{}'", symbol)
                return None
                
            # Sort by expiry date to find the nearest month active contract
            # expiry is usually in YYYY-MM-DD or date format, or None
            def parse_expiry(item):
                exp = item[2]
                if not exp:
                    return datetime.date.max
                if isinstance(exp, str):
                    try:
                        return datetime.datetime.strptime(exp, "%Y-%m-%d").date()
                    except ValueError:
                        return datetime.date.max
                return exp
                
            futures.sort(key=parse_expiry)
            
            # Select the nearest active contract
            active_key, active_symbol, active_expiry = futures[0]
            logger.info(
                "Resolved active futures contract for {}: Symbol={}, Key={}, Expiry={}",
                symbol, active_symbol, active_key, active_expiry
            )
            
            self._key_cache[symbol] = active_key
            return active_key
            
        except ApiException as e:
            logger.error("ApiException during contract resolution for {}: {}", symbol, e)
            return None
        except Exception as e:
            logger.error("Error during contract resolution for {}: {}", symbol, e)
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all active short-term and open positions.
        
        Returns
        -------
        positions : list of dicts
            List of position details from Upstox.
        """
        url = 'https://api.upstox.com/v2/portfolio/short-term-positions'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                res_data = response.json()
                return res_data.get('data', [])
            else:
                logger.error("Failed to fetch positions: Status={}, Response={}", response.status_code, response.text)
                return []
        except Exception as e:
            logger.error("Error fetching positions: {}", e)
            return []

    def get_position_for_asset(self, instrument_key: str) -> Optional[Dict[str, Any]]:
        """Get position details for a specific instrument contract key."""
        positions = self.get_positions()
        for pos in positions:
            if pos.get('instrument_key') == instrument_key:
                return pos
        return None

    def place_mcx_order(
        self,
        symbol: str,
        transaction_type: str,  # 'BUY' or 'SELL'
        quantity: int,          # Number of lots
        order_type: str = "MARKET",
        price: float = 0.0,
        product: str = "D"      # 'D' for NRML delivery, 'I' for MIS intraday
    ) -> Optional[str]:
        """Place an order for MCX futures contract.
        
        Parameters
        ----------
        symbol : str
            Base symbol (e.g. 'GOLD', 'CRUDEOIL') or full instrument key.
        transaction_type : str
            'BUY' or 'SELL'.
        quantity : int
            Lots to trade.
        order_type : str
            'MARKET', 'LIMIT', 'SL', 'SL-M'.
        price : float
            Limit price.
        product : str
            'D' (NRML/Delivery) or 'I' (MIS/Intraday).
            
        Returns
        -------
        order_id : str or None
            Unique Upstox order ID if successful, otherwise None.
        """
        # 1. Resolve active contract if symbol is given instead of a raw key
        instrument_key = symbol
        if "|" not in symbol:
            resolved = self.resolve_active_future_contract(symbol)
            if not resolved:
                logger.error("Could not resolve futures key for symbol '{}'. Aborting order.", symbol)
                return None
            instrument_key = resolved
            
        # 2. Build PlaceOrderRequest
        body = upstox_client.PlaceOrderRequest(
            quantity=int(quantity),
            product=product,
            validity='DAY',
            price=float(price) if order_type == "LIMIT" else 0.0,
            instrument_token=instrument_key,
            order_type=order_type,
            transaction_type=transaction_type,
            disclosed_quantity=0,
            trigger_price=0.0,
            is_amo=False
        )
        
        try:
            logger.info(
                "Submitting MCX {} order: Key={}, Qty={}, Type={}, Product={}",
                transaction_type, instrument_key, quantity, order_type, product
            )
            api_response = self.order_api.place_order(body, api_version='2.0')
            order_id = api_response.data.order_id
            logger.success("Order placed successfully. ID={}", order_id)
            return order_id
        except ApiException as e:
            logger.error("ApiException placing Upstox order: {}", e.body if hasattr(e, 'body') else e)
            return None
        except Exception as e:
            logger.error("Unexpected error placing Upstox order: {}", e)
            return None

    def execute_rl_action(
        self,
        asset_config: AssetConfig,
        action: float,       # Continuous action in range [-1.0, 1.0]
        portfolio_value: float
    ) -> Optional[str]:
        """Translates a continuous reinforcement learning action [-1, 1] into an execution order.
        
        - Positive action (e.g. 0.8) indicates we want to be LONG (80% allocation)
        - Negative action (e.g. -0.5) indicates we want to be SHORT (50% allocation)
        - 0.0 action indicates we want to be flat (0% allocation)
        
        Parameters
        ----------
        asset_config : AssetConfig
            The configuration dataclass for the asset.
        action : float
            Target continuous allocation [-1, 1].
        portfolio_value : float
            Current total portfolio value (INR) to size the position.
        """
        # 1. Resolve contract
        base_symbol = asset_config.symbol
        instrument_key = self.resolve_active_future_contract(base_symbol)
        if not instrument_key:
            logger.error("Upstox resolution failed for {}. Order aborted.", base_symbol)
            return None
            
        # 2. Get current position details
        position = self.get_position_for_asset(instrument_key)
        
        # Upstox net_quantity represents current shares/lots held (positive = long, negative = short)
        current_qty = int(position.get('net_quantity', 0)) if position else 0
        
        # 3. Calculate target position sizing in lots
        # Contract Value = price * lot_size
        # Since we don't have price directly here, we make a quick quote request
        price = self.get_last_price(instrument_key)
        if price <= 0:
            logger.error("Could not fetch current price for {}. Sizing aborted.", base_symbol)
            return None
            
        lot_value = price * asset_config.lot_size
        
        # Max allowable lots based on portfolio value allocation
        target_allocation_value = portfolio_value * action
        target_lots = int(target_allocation_value / lot_value)
        
        # Lot trade difference
        trade_diff = target_lots - current_qty
        
        if trade_diff == 0:
            logger.info("Current position ({}) matches target lots ({}) for {}. No action taken.", current_qty, target_lots, base_symbol)
            return None
            
        # Determine trade parameters
        tx_type = "BUY" if trade_diff > 0 else "SELL"
        trade_qty = abs(trade_diff)
        
        logger.info(
            "Target allocation for {}: Action={:.4f}, TargetLots={}, CurrentLots={}, TradeQty={}, Action={}",
            base_symbol, action, target_lots, current_qty, trade_qty, tx_type
        )
        
        return self.place_mcx_order(
            symbol=instrument_key,
            transaction_type=tx_type,
            quantity=trade_qty,
            order_type="MARKET",
            product="D"  # NRML positional futures
        )

    def get_last_price(self, instrument_key: str) -> float:
        """Fetch current LTP (Last Traded Price) for an instrument."""
        url = f'https://api.upstox.com/v2/market-quote/ohlc'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        params = {'symbol': instrument_key}
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json().get('data', {})
                inst_data = data.get(instrument_key, {})
                ohlc = inst_data.get('ohlc', {})
                return float(ohlc.get('close', 0.0))  # close holds latest LTP in ohlc api
            else:
                logger.error("Failed to fetch LTP: Status={}, Text={}", response.status_code, response.text)
                return 0.0
        except Exception as e:
            logger.error("Error fetching LTP: {}", e)
            return 0.0

    def close_all_positions(self) -> bool:
        """Exit all positions held in the account."""
        url = 'https://api.upstox.com/v2/order/positions/exit'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        try:
            logger.warning("UpstoxExecutor: Requesting exit of ALL positions.")
            response = requests.post(url, json={}, headers=headers)
            if response.status_code == 200:
                logger.success("Exit all positions request successful.")
                return True
            else:
                logger.error("Failed to exit positions: Status={}, Text={}", response.status_code, response.text)
                return False
        except Exception as e:
            logger.error("Error exiting positions: {}", e)
            return False
