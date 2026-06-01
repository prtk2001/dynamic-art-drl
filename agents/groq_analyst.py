"""
Dynamic ART-DRL — Groq LLM Meta-Analyst
==========================================
Advisory AI analyst powered by Groq's Llama 3.3 70B model.

This module provides **qualitative market commentary and risk
assessments**.  It does NOT make buy/sell decisions — those come
exclusively from the DRL agents.

Capabilities:
    • ``analyze_market_state``  — natural-language read on current state
    • ``generate_trade_commentary`` — explain *why* a trade was taken
    • ``assess_risk``           — structured risk score + commentary
    • ``post_session_report``   — end-of-session performance narrative

Uses Groq's tool/function-calling interface for structured JSON output
where appropriate.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from config import settings

# ── Lazy import of groq SDK (fail gracefully if not installed) ───────────
try:
    from groq import Groq, APIError, RateLimitError
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    logger.warning("groq SDK not installed — GroqAnalyst will be unavailable")


# ═════════════════════════════════════════════════════════════════════════════
# Tool / Function Definitions (for structured output)
# ═════════════════════════════════════════════════════════════════════════════

_RISK_ASSESSMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_risk_assessment",
        "description": (
            "Submit a structured risk assessment for the current portfolio "
            "state and market regime."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "risk_score": {
                    "type": "number",
                    "description": (
                        "Numerical risk score from 0 (very safe) to 10 "
                        "(extremely risky)."
                    ),
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "moderate", "high", "extreme"],
                    "description": "Categorical risk level.",
                },
                "key_risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Top 3-5 specific risk factors identified.",
                },
                "recommendation": {
                    "type": "string",
                    "description": (
                        "Brief advisory recommendation (e.g. reduce exposure, "
                        "maintain position, hedge)."
                    ),
                },
                "commentary": {
                    "type": "string",
                    "description": "Detailed narrative risk commentary.",
                },
            },
            "required": [
                "risk_score",
                "risk_level",
                "key_risks",
                "recommendation",
                "commentary",
            ],
        },
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# GroqAnalyst
# ═════════════════════════════════════════════════════════════════════════════

class GroqAnalyst:
    """Groq-powered LLM meta-analyst for advisory market commentary.

    This class is intentionally **not** a ``BaseAgent`` subclass — it
    provides qualitative overlays, not actionable trading signals.

    Parameters
    ----------
    api_key : str | None
        Groq API key.  Defaults to ``settings.GROQ_API_KEY``.
    model : str
        Model identifier.  Defaults to ``settings.GROQ_MODEL``.
    max_retries : int
        Maximum retry attempts on transient API errors.
    retry_delay : float
        Base delay (seconds) between retries (exponential back-off).
    temperature : float
        Sampling temperature for generation.
    max_tokens : int
        Maximum tokens per response.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = settings.GROQ_MODEL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        if not _GROQ_AVAILABLE:
            raise ImportError(
                "groq SDK is required. Install with: pip install groq"
            )

        self._api_key = api_key or settings.GROQ_API_KEY
        if not self._api_key:
            raise ValueError(
                "Groq API key not set. Provide via constructor or GROQ_API_KEY env var."
            )

        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = Groq(api_key=self._api_key)

        logger.info(
            "GroqAnalyst initialised | model={} temp={} max_tokens={}",
            model, temperature, max_tokens,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def analyze_market_state(self, state_dict: Dict[str, Any]) -> str:
        """Generate a natural-language market state analysis.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing current market data, e.g.::

                {
                    "asset": "GLD",
                    "price": 185.42,
                    "returns_1d": 0.003,
                    "volatility_30d": 0.14,
                    "rsi_14": 62.3,
                    "macd_signal": "bullish_cross",
                    "kalman_trend": "upward",
                    "regime": "trending",
                }

        Returns
        -------
        str
            Natural-language analysis paragraph(s).
        """
        system_prompt = (
            "You are a senior quantitative commodity analyst. "
            "Given the market state data below, provide a concise but "
            "insightful analysis covering: current trend, momentum, "
            "volatility regime, and any notable signals.  Be specific "
            "about the numbers.  Do NOT recommend trades — only analyze."
        )
        user_prompt = (
            f"Current market state:\n```json\n"
            f"{json.dumps(state_dict, indent=2, default=str)}\n```\n\n"
            "Provide your analysis."
        )
        return self._chat(system_prompt, user_prompt)

    def generate_trade_commentary(
        self,
        trade_action: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> str:
        """Explain why a particular trade was executed.

        Parameters
        ----------
        trade_action : dict
            Details of the trade, e.g.::

                {
                    "agent": "PPO",
                    "action": 0.72,
                    "direction": "buy",
                    "shares": 45,
                    "value": 8344.0,
                }

        market_context : dict
            Current market conditions (same schema as ``analyze_market_state``).

        Returns
        -------
        str
            Commentary explaining the trade rationale.
        """
        system_prompt = (
            "You are a trade commentary analyst for an algorithmic trading "
            "system.  Given the DRL agent's trade action and current market "
            "context, provide a brief (2-4 sentence) commentary explaining "
            "the likely rationale behind the trade.  You are NOT endorsing "
            "or recommending the trade — just explaining what the agent "
            "might be reacting to."
        )
        user_prompt = (
            f"Trade action:\n```json\n"
            f"{json.dumps(trade_action, indent=2, default=str)}\n```\n\n"
            f"Market context:\n```json\n"
            f"{json.dumps(market_context, indent=2, default=str)}\n```\n\n"
            "Provide your commentary."
        )
        return self._chat(system_prompt, user_prompt)

    def assess_risk(
        self,
        portfolio_state: Dict[str, Any],
        regime: str,
    ) -> Dict[str, Any]:
        """Produce a structured risk assessment using function calling.

        Parameters
        ----------
        portfolio_state : dict
            Portfolio metrics, e.g.::

                {
                    "portfolio_value": 104_250,
                    "cash": 22_100,
                    "shares_held": 445,
                    "unrealised_pnl": 4250,
                    "drawdown": 0.02,
                    "sharpe_30d": 1.4,
                    "sortino_30d": 1.9,
                }

        regime : str
            Current volatility regime: "low_vol", "trending", "high_vol",
            or "crisis".

        Returns
        -------
        dict
            Structured risk assessment with keys:
            ``risk_score``, ``risk_level``, ``key_risks``,
            ``recommendation``, ``commentary``.
        """
        system_prompt = (
            "You are a risk management analyst for a commodity trading desk. "
            "Assess the risk of the current portfolio given the market regime. "
            "Use the submit_risk_assessment function to provide your structured "
            "assessment."
        )
        user_prompt = (
            f"Portfolio state:\n```json\n"
            f"{json.dumps(portfolio_state, indent=2, default=str)}\n```\n\n"
            f"Current market regime: **{regime}**\n\n"
            "Provide your risk assessment using the tool."
        )
        return self._chat_with_tools(
            system_prompt,
            user_prompt,
            tools=[_RISK_ASSESSMENT_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_risk_assessment"}},
        )

    def post_session_report(self, session_stats: Dict[str, Any]) -> str:
        """Generate a comprehensive end-of-session performance report.

        Parameters
        ----------
        session_stats : dict
            Session-level statistics, e.g.::

                {
                    "asset": "GLD",
                    "timeframe": "1d",
                    "agent": "PPO",
                    "total_return_pct": 4.25,
                    "sharpe_ratio": 1.42,
                    "sortino_ratio": 1.88,
                    "max_drawdown_pct": 3.1,
                    "total_trades": 87,
                    "win_rate_pct": 58.6,
                    "avg_trade_pnl": 48.7,
                    "total_transaction_costs": 312.5,
                    "training_timesteps": 100_000,
                    "regime_distribution": {
                        "low_vol": 0.35,
                        "trending": 0.40,
                        "high_vol": 0.20,
                        "crisis": 0.05,
                    },
                }

        Returns
        -------
        str
            Multi-paragraph performance report.
        """
        system_prompt = (
            "You are a portfolio performance analyst writing an end-of-session "
            "report for stakeholders.  Given the session statistics, produce a "
            "structured report covering:\n"
            "1. Performance Summary (return, Sharpe, drawdown)\n"
            "2. Trading Activity (trade count, win rate, costs)\n"
            "3. Risk Analysis (drawdown, regime exposure)\n"
            "4. Key Observations & Recommendations\n\n"
            "Use precise numbers from the data.  Keep it professional and concise."
        )
        user_prompt = (
            f"Session statistics:\n```json\n"
            f"{json.dumps(session_stats, indent=2, default=str)}\n```\n\n"
            "Generate the performance report."
        )
        return self._chat(system_prompt, user_prompt)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a simple chat completion request with retries.

        Returns the assistant's text content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                content = response.choices[0].message.content
                logger.debug(
                    "Groq chat OK | model={} tokens_used={}",
                    self._model,
                    response.usage.total_tokens if response.usage else "?",
                )
                return content or ""

            except RateLimitError as exc:
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Groq rate-limit hit (attempt {}/{}). Waiting {:.1f}s",
                    attempt, self._max_retries, wait,
                )
                time.sleep(wait)

            except APIError as exc:
                logger.error("Groq API error: {}", exc)
                if attempt == self._max_retries:
                    raise
                time.sleep(self._retry_delay)

            except Exception as exc:
                logger.error("Unexpected error calling Groq: {}", exc)
                raise

        return "[GroqAnalyst] Failed to get response after retries."

    def _chat_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion with function/tool calling.

        Extracts and parses the tool call arguments as a dict.
        Falls back to an error dict on failure.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                }
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice

                response = self._client.chat.completions.create(**kwargs)
                message = response.choices[0].message

                # Extract tool call arguments
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    arguments = json.loads(tool_call.function.arguments)
                    logger.debug(
                        "Groq tool call OK | function={} tokens={}",
                        tool_call.function.name,
                        response.usage.total_tokens if response.usage else "?",
                    )
                    return arguments

                # Fallback: model returned text instead of tool call
                logger.warning(
                    "Groq returned text instead of tool call. "
                    "Returning as commentary."
                )
                return {
                    "risk_score": -1,
                    "risk_level": "unknown",
                    "key_risks": [],
                    "recommendation": "Unable to produce structured assessment.",
                    "commentary": message.content or "",
                }

            except RateLimitError:
                wait = self._retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Groq rate-limit (attempt {}/{}). Waiting {:.1f}s",
                    attempt, self._max_retries, wait,
                )
                time.sleep(wait)

            except (APIError, json.JSONDecodeError) as exc:
                logger.error("Groq tool-call error: {}", exc)
                if attempt == self._max_retries:
                    return {
                        "risk_score": -1,
                        "risk_level": "error",
                        "key_risks": [str(exc)],
                        "recommendation": "Error — manual review required.",
                        "commentary": f"API error after {self._max_retries} retries: {exc}",
                    }
                time.sleep(self._retry_delay)

            except Exception as exc:
                logger.error("Unexpected error in tool call: {}", exc)
                raise

        return {
            "risk_score": -1,
            "risk_level": "error",
            "key_risks": ["Max retries exceeded"],
            "recommendation": "Error — manual review required.",
            "commentary": "Failed to get response from Groq API.",
        }

    def __repr__(self) -> str:
        return (
            f"GroqAnalyst(model='{self._model}', "
            f"temp={self._temperature}, max_tokens={self._max_tokens})"
        )
