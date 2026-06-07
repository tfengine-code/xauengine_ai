import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import OpenAI

app = FastAPI(title="XAUUSD AI Trading Engine")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
    base_url="https://ai.sumopod.com/v1"
)

class HistoryCandle(BaseModel):
    o: float
    h: float
    l: float
    c: float

class AdvancedSignalPayload(BaseModel):
    symbol: str
    timeframe: str
    signal: str
    current_price: float
    vwap_price: float
    supertrend_h: float
    supertrend_l: float
    msh: float
    msl: float
    atr_14: float
    atr_50: float
    range_1h: float
    range_4h: float
    velocity: float
    body_size: float
    body_ratio: float
    momentum_5: float
    day_high: float
    day_low: float
    dist_dh: float
    dist_dl: float
    spread: float
    session: str
    history_candles: List[HistoryCandle]

class AIResponse(BaseModel):
    confidence: int
    approved: bool
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    cancel_pending: bool

@app.get("/")
def read_root():
    return {"status": "AI Engine Running", "version": "2.0"}

@app.post("/analyze_signal", response_model=AIResponse)
def analyze_signal(payload: AdvancedSignalPayload):
    try:
        # Construct the advanced institutional prompt
        history_summary = "[\n"
        for idx, candle in enumerate(payload.history_candles[-10:]):  # Just list the last 10 in string directly, but model gets context of volatility
            history_summary += f"  {{O: {candle.o}, H: {candle.h}, L: {candle.l}, C: {candle.c}}},\n"
        history_summary += "]"

        prompt = f"""
        You are an elite Institutional AI Trader specializing in XAUUSD (Gold).
        Analyze the following comprehensive M1 market snapshot before confirming the setup.
        
        [Setup Direction]: {payload.signal}
        
        [Trend & Structure]
        - Current Price: {payload.current_price}
        - Daily VWAP: {payload.vwap_price}
        - SuperTrend Resistance (H): {payload.supertrend_h}
        - SuperTrend Support (L): {payload.supertrend_l}
        - Major Market Structure High (MSH): {payload.msh}
        - Major Market Structure Low (MSL): {payload.msl}
        
        [Volatility & Range]
        - ATR(14): {payload.atr_14}
        - ATR(50): {payload.atr_50}
        - 1H Range (points): {payload.range_1h}
        - 4H Range (points): {payload.range_4h}
        
        [Momentum & Price Action]
        - Velocity (points/min): {payload.velocity}
        - Latest Candle Body Size: {payload.body_size}
        - Body/Wick Ratio: {payload.body_ratio}
        - Momentum (last 5 candles net): {payload.momentum_5}
        
        [Market Context]
        - Session: {payload.session}
        - Day High / Day Low: {payload.day_high} / {payload.day_low}
        - Distance to Day High: {payload.dist_dh}
        - Distance to Day Low: {payload.dist_dl}
        - Current Spread: {payload.spread}
        
        [Recent Price Action Memory (Last 10 of 50 candles)]:
        {history_summary}
        
        Your task is to analyze this data considering standard technical analysis, retracement strategies, momentum validation, and macroeconomic sentiment (using your GPT memory for general current conditions).
        
        Rules:
        - We only want high-probability trades (Confidence > 75).
        - If the setup probability is extremely high and the current price is already at an optimal level, use 'Buy' or 'Sell' for direct market execution.
        - Otherwise, calculate an optimal retracement entry using SuperTrend bounds or MSH/MSL.
        - Stop Loss MUST be safe, placed outside the SuperTrend H/L bounds or Major Structure bounds.
        - Take Profit should maximize RR, aiming for at least 1:2. The EA will handle Smart Breakeven at 1.25R.
        - If confidence is low due to poor momentum or structure, set approved to false.
        
        Provide a JSON response (without markdown blocks) with the following structure:
        {{
            "confidence": <int 0-100>,
            "approved": <boolean>,
            "action": "<string: 'Buy', 'Sell', 'Buy Limit', 'Sell Limit', 'Buy Stop', 'Sell Stop'>",
            "entry_price": <float>,
            "stop_loss": <float>,
            "take_profit": <float>,
            "cancel_pending": <boolean>
        }}
        """

        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": "You are a specialized JSON-only institutional trading AI. Always return valid JSON matching the schema."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        data = json.loads(content)
        return AIResponse(**data)
        
    except Exception as e:
        print(f"Error during AI analysis: {e}")
        return AIResponse(
            confidence=0,
            approved=False,
            action="None",
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            cancel_pending=False
        )
