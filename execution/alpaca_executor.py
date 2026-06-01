"""
Dynamic ART-DRL — Alpaca Paper Trading Execution Layer
======================================================
Connects to the Alpaca Paper Trading API v2 using the official alpaca-py SDK.
Translates reinforcement learning actions into buy/sell stock orders for commodity ETFs
(GLD and USO).
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from loguru import logger
from typing import Optional, Dict, Any, List

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_ENDPOINT
)
from config.assets import AssetConfig


class AlpacaExecutor:
    """Handles order placement, position query, and account state checks via Alpaca API."""
    
    def __init__(self) -> None:
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.endpoint = ALPACA_ENDPOINT
        
        # Initialize client in paper-trading mode
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=True
        )
        logger.info("AlpacaExecutor initialised. Endpoint={}", self.endpoint)
        
    def get_account_info(self) -> Dict[str, float]:
        """Fetch total portfolio value, buying power, and available cash.
        
        Returns
        -------
        info : dict
            Contains 'cash', 'portfolio_value', and 'buying_power'.
        """
        try:
            account = self.trading_client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power)
            }
        except Exception as e:
            logger.error("Error fetching Alpaca account: {}", e)
            return {"cash": 0.0, "portfolio_value": 0.0, "buying_power": 0.0}

    def get_positions(self) -> Dict[str, int]:
        """Get current shares held for each symbol.
        
        Returns
        -------
        positions : dict
            Mapping of symbol string to quantity integer (positive=long, negative=short).
        """
        try:
            positions = self.trading_client.get_all_positions()
            pos_dict = {}
            for pos in positions:
                # qty is string, cast to int
                qty = int(pos.qty)
                if pos.side == 'short':
                    qty = -qty
                pos_dict[pos.symbol] = qty
            return pos_dict
        except Exception as e:
            logger.error("Error fetching Alpaca positions: {}", e)
            return {}

    def get_position_for_symbol(self, symbol: str) -> int:
        """Get shares held for a specific symbol."""
        positions = self.get_positions()
        return positions.get(symbol, 0)

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        time_in_force: TimeInForce = TimeInForce.GTC
    ) -> Optional[str]:
        """Submit a market order to Alpaca paper endpoint.
        
        Parameters
        ----------
        symbol : str
            ETF ticker symbol (e.g. GLD, USO).
        side : OrderSide
            BUY or SELL.
        qty : int
            Number of shares.
        time_in_force : TimeInForce
            GTC (Good till Cancelled) or DAY.
            
        Returns
        -------
        order_id : str or None
            Alpaca order ID if submitted successfully, otherwise None.
        """
        if qty <= 0:
            logger.warning("Order quantity is zero or negative ({}). Aborting.", qty)
            return None
            
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=time_in_force
        )
        
        try:
            logger.info("Submitting Alpaca market order: {} {} shares of {}", side.value, qty, symbol)
            order = self.trading_client.submit_order(order_data=order_request)
            logger.success("Order submitted successfully. ID={}, Status={}", order.id, order.status)
            return str(order.id)
        except Exception as e:
            logger.error("Error submitting Alpaca order: {}", e)
            return None

    def execute_rl_action(
        self,
        asset_config: AssetConfig,
        action: float,       # Continuous action [-1.0, 1.0]
        portfolio_value: float
    ) -> Optional[str]:
        """Translate continuous action [-1, 1] into an Alpaca buy/sell order.
        
        Parameters
        ----------
        asset_config : AssetConfig
            Asset config parameters.
        action : float
            Continuous value representing target allocation percentage.
        portfolio_value : float
            Total account portfolio value in USD.
        """
        symbol = asset_config.symbol
        
        # 1. Fetch current price to size position
        # For simplicity, we can get the current market price of the asset
        # (This can also be fetched using historical data client or quotes API,
        # but here we query Alpaca's current position valuation to find the price,
        # or fall back to an estimate)
        price = self._get_asset_price_estimate(symbol)
        if price <= 0:
            logger.error("Could not fetch price estimate for {}. Order aborted.", symbol)
            return None
            
        # 2. Get current shares held
        current_qty = self.get_position_for_symbol(symbol)
        
        # 3. Size target quantity
        target_allocation_value = portfolio_value * action
        target_qty = int(target_allocation_value / price)
        
        # Trade difference
        trade_diff = target_qty - current_qty
        
        if trade_diff == 0:
            logger.info("Current shares ({}) matches target ({}) for {}. No action.", current_qty, target_qty, symbol)
            return None
            
        side = OrderSide.BUY if trade_diff > 0 else OrderSide.SELL
        trade_qty = abs(trade_diff)
        
        logger.info(
            "Target allocation for {}: Action={:.4f}, TargetShares={}, CurrentShares={}, TradeQty={}, Side={}",
            symbol, action, target_qty, current_qty, trade_qty, side.value
        )
        
        return self.place_order(symbol=symbol, side=side, qty=trade_qty)

    def _get_asset_price_estimate(self, symbol: str) -> float:
        """Fetch current valuation price of an asset, fallback if no position."""
        try:
            # Check if we already hold a position, which has the current price
            position = self.trading_client.get_open_position(symbol_or_asset_id=symbol)
            return float(position.current_price)
        except Exception:
            # Fallback: Query a single day's bar or use a hardcoded default for safety
            # A robust production implementation would query historical data
            defaults = {"GLD": 220.0, "USO": 75.0}
            logger.warning("Could not find open position price for {}. Using default fallback.", symbol)
            return defaults.get(symbol, 100.0)

    def close_all_positions(self) -> bool:
        """Liquidate all open positions in the account."""
        try:
            logger.warning("AlpacaExecutor: Requesting exit of ALL positions.")
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.success("Liquidation requests submitted successfully.")
            return True
        except Exception as e:
            logger.error("Error liquidating Alpaca portfolio: {}", e)
            return False
