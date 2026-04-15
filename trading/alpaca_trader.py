"""
Alpaca paper trading integration.

Submits market orders when the pipeline fires BUY/STRONG_BUY signals,
and closes positions when SELL/STRONG_SELL signals fire.

Always uses Alpaca's paper trading endpoint — never touches real money.

Setup:
    1. Sign up at alpaca.markets (free)
    2. Go to Paper Trading → API Keys → generate key pair
    3. Add to .env:
           ALPACA_API_KEY=your_key_id
           ALPACA_SECRET_KEY=your_secret
           PAPER_TRADING_ENABLED=true
           POSITION_SIZE_USD=1000

Install SDK:
    venv/bin/pip install alpaca-py
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"

# Labels that trigger a buy order
BUY_LABELS = {"STRONG_BUY", "BUY"}
# Labels that trigger closing an open position
SELL_LABELS = {"STRONG_SELL", "SELL"}


@dataclass
class TradeResult:
    ticker: str
    action: str            # "buy" | "sell" | "close" | "skipped" | "error"
    qty: float | None
    price: float | None
    order_id: str | None
    position_size_usd: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.action not in ("skipped", "error")


class AlpacaTrader:
    """
    Thin wrapper around alpaca-py for paper trading.

    Usage:
        trader = AlpacaTrader()
        if trader.enabled:
            result = trader.handle_signal("AAPL", "BUY", 0.74, signal_id=1)
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        position_size_usd: float = None,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.position_size_usd = position_size_usd or float(
            os.getenv("POSITION_SIZE_USD", "1000")
        )
        self.enabled = (
            os.getenv("PAPER_TRADING_ENABLED", "false").lower() == "true"
            and bool(self.api_key)
            and bool(self.secret_key)
        )
        self._client = None

        if self.enabled:
            logger.info(
                "AlpacaTrader enabled (paper) position_size=$%.0f",
                self.position_size_usd,
            )
        else:
            logger.info("AlpacaTrader disabled (set PAPER_TRADING_ENABLED=true to enable)")

    def _get_client(self):
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError:
                raise ImportError(
                    "alpaca-py not installed. Run: venv/bin/pip install alpaca-py"
                )
            self._client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=True,
            )
        return self._client

    def handle_signal(
        self,
        ticker: str,
        label: str,
        confluence_score: float,
        signal_id: int = None,
    ) -> TradeResult:
        """
        Act on a confluence signal.

        BUY/STRONG_BUY  → open a long position (if not already open)
        SELL/STRONG_SELL → close existing position (if open)
        HOLD             → do nothing

        Returns TradeResult describing what happened.
        """
        if not self.enabled:
            return TradeResult(
                ticker=ticker, action="skipped", qty=None,
                price=None, order_id=None, position_size_usd=None,
                error="Paper trading disabled",
            )

        if label in BUY_LABELS:
            return self._open_position(ticker, label, confluence_score, signal_id)
        elif label in SELL_LABELS:
            return self._close_position(ticker, label, signal_id)
        else:
            return TradeResult(
                ticker=ticker, action="skipped", qty=None,
                price=None, order_id=None, position_size_usd=None,
            )

    def _open_position(
        self, ticker: str, label: str, confluence_score: float, signal_id: int
    ) -> TradeResult:
        """Submit a notional market buy order."""
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            client = self._get_client()

            # Check if we already have a position
            try:
                existing = client.get_open_position(ticker)
                if existing:
                    logger.info("%s: position already open (%s shares), skipping buy",
                                ticker, existing.qty)
                    return TradeResult(
                        ticker=ticker, action="skipped", qty=None,
                        price=None, order_id=None, position_size_usd=None,
                    )
            except Exception:
                pass  # No existing position — proceed

            order_data = MarketOrderRequest(
                symbol=ticker,
                notional=self.position_size_usd,   # dollar amount, Alpaca calculates shares
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )

            order = client.submit_order(order_data)
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None
            qty = float(order.filled_qty) if order.filled_qty else None

            logger.info(
                "%s: BUY order submitted (id=%s label=%s confluence=%.3f size=$%.0f)",
                ticker, order.id, label, confluence_score, self.position_size_usd,
            )

            return TradeResult(
                ticker=ticker,
                action="buy",
                qty=qty,
                price=fill_price,
                order_id=str(order.id),
                position_size_usd=self.position_size_usd,
            )

        except Exception as e:
            logger.error("%s: buy order failed: %s", ticker, e)
            return TradeResult(
                ticker=ticker, action="error", qty=None,
                price=None, order_id=None, position_size_usd=None,
                error=str(e),
            )

    def _close_position(self, ticker: str, label: str, signal_id: int) -> TradeResult:
        """Close an existing position."""
        try:
            client = self._get_client()

            # Check if we have a position to close
            try:
                position = client.get_open_position(ticker)
            except Exception:
                logger.info("%s: no open position to close", ticker)
                return TradeResult(
                    ticker=ticker, action="skipped", qty=None,
                    price=None, order_id=None, position_size_usd=None,
                )

            qty = float(position.qty)
            order = client.close_position(ticker)
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None

            logger.info(
                "%s: CLOSE order submitted (id=%s label=%s qty=%.4f)",
                ticker, order.id, label, qty,
            )

            return TradeResult(
                ticker=ticker,
                action="close",
                qty=qty,
                price=fill_price,
                order_id=str(order.id),
                position_size_usd=None,
            )

        except Exception as e:
            logger.error("%s: close order failed: %s", ticker, e)
            return TradeResult(
                ticker=ticker, action="error", qty=None,
                price=None, order_id=None, position_size_usd=None,
                error=str(e),
            )

    def get_portfolio_summary(self) -> dict:
        """
        Return a summary of open positions and account equity.
        Used by the dashboard.
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            client = self._get_client()
            account = client.get_account()
            positions = client.get_all_positions()

            return {
                "enabled": True,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "positions": [
                    {
                        "ticker": p.symbol,
                        "qty": float(p.qty),
                        "market_value": float(p.market_value),
                        "unrealized_pnl": float(p.unrealized_pl),
                        "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                    }
                    for p in positions
                ],
            }
        except Exception as e:
            logger.error("Failed to fetch portfolio summary: %s", e)
            return {"enabled": True, "error": str(e)}
