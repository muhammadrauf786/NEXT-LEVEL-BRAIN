#!/usr/bin/env python3
"""
NEXT LEVEL BRAIN - Live Trading System (CLI only)
All-in-one live trading with AI enhancement
Created by: Aleem Shahzad | AI Partner: Claude (Anthropic)
"""

import asyncio
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import sys
import signal
import threading
import time
from typing import Dict, List, Optional, Tuple
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("logs/live_trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

class TradingBrain:
    """AI Trading Brain with Neural Network"""
    
    def __init__(self):
        self.memories = []
        self.model_trained = False
        self.confidence_threshold = 0.6
        self.sentiment_decision = "ALLOW"
        self.risk_modifier = 1.0
        self._load_memories()
        self._check_sentiment_bias()

    def _check_sentiment_bias(self):
        """Load external market intelligence report"""
        try:
            report_file = Path("latest_intelligence_report.txt")
            if report_file.exists():
                with open(report_file, 'r') as f:
                    content = f.read()
                    if "DECISION:           BLOCK" in content:
                        self.sentiment_decision = "BLOCK"
                    elif "DECISION:           REDUCE" in content:
                        self.sentiment_decision = "REDUCE"
                        self.risk_modifier = 0.5
                    else:
                        self.sentiment_decision = "ALLOW"
                        self.risk_modifier = 1.0
                logger.info(f"📡 Market Intelligence: {self.sentiment_decision}")
            else:
                logger.warning("📡 No intelligence report found. Defaulting to ALLOW.")
        except Exception as e:
            logger.error(f"Failed to read sentiment report: {e}")

    def _load_memories(self):
        """Load trained memories from file"""
        try:
            memory_file = Path("models/ai_memories.json")
            if memory_file.exists():
                import json
                with open(memory_file, 'r') as f:
                    self.memories = json.load(f)
                self.model_trained = True
                logger.info(f"🧠 Loaded {len(self.memories)} training memories. AI is ready.")
            else:
                logger.warning("⚠️ No training data found. AI starting with blank slate.")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")

    def _is_silver_bullet_time(self, timestamp: datetime) -> bool:
        """
        Check if time is within ICT Silver Bullet windows (EST based).
        Windows: 3-4 AM (London), 10-11 AM (NY AM), 2-3 PM (NY PM).
        
        We assume NY time for these windows.
        """
        # If user has not configured timezone, we assume current system time is NY or MT5 Server time.
        # To be safe, we check 'MT5_SERVER_TIME_OFFSET' if defined in .env
        offset = int(os.getenv("MT5_SERVER_TIME_OFFSET", 0))
        adj_time = timestamp + timedelta(hours=offset)
        h = adj_time.hour
        
        if h in [3, 10, 14]:
            return True
        return False

    def analyze_market(self, symbol: str, data: pd.DataFrame) -> Dict:
        """ICT/SMC AI market analysis"""
        try:
            if len(data) < 50:
                return {'action': 'HOLD', 'bias': 'NEUTRAL', 'confidence': 0.0, 'reasoning': 'Insufficient data'}
            
            # Add technical indicators
            data = self._add_indicators(data)
            index = len(data) - 1  # Current bar index
            
            # ALWAYS determine market bias for systems like Grid
            market_bias = self._determine_market_bias(data, index)
            
            # Check Silver Bullet Time (Only blocks ICT execution, not bias detection)
            current_time = datetime.now()
            is_sb_time = self._is_silver_bullet_time(current_time)
            
            # Update sentiment from file regularly
            self._check_sentiment_bias()
            
            if self.sentiment_decision == "BLOCK":
                return {'action': 'HOLD', 'bias': market_bias, 'confidence': 0.0, 'reasoning': 'Intelligence Engine Decision: BLOCK'}

            # ICT Signal filtering
            if not is_sb_time:
                 # We still return the bias so Grid can work, but signal is HOLD for ICT
                 return {'action': 'HOLD', 'bias': market_bias, 'confidence': 0.0, 'reasoning': 'Outside Silver Bullet Windows'}
            
            # Check for liquidity sweeps
            liquidity_sweep = self._detect_liquidity_sweep(data, index)
            
            # Check for displacement and FVG
            fvg = self._detect_fair_value_gap(data, index)
            
            # Check dealing range and discount/premium zones
            dealing_range = self._analyze_dealing_range(data, index)
            
            # Check for order blocks
            order_block = self._detect_order_block(data, index, market_bias)
            
            # Check OTE (Optimal Trade Entry) levels
            ote_level = self._check_ote_levels(data, index, fvg)
            
            current = data.iloc[index]
            
            # Calculate confluence score for available signals
            signals_present = []
            signal_strengths = []
            
            if liquidity_sweep['detected']:
                signals_present.append(f"Liquidity Sweep ({liquidity_sweep['type']})")
                signal_strengths.append(liquidity_sweep['strength'])
            
            if fvg['detected']:
                signals_present.append(f"FVG ({fvg['type']})")
                signal_strengths.append(fvg['strength'])
            
            if order_block['detected']:
                signals_present.append(f"Order Block ({order_block['type']})")
                signal_strengths.append(order_block['strength'])
            
            # BULLISH SETUP - Need at least 2 of 3 main signals
            if market_bias == 'BULLISH':
                bullish_conditions = 0
                
                if liquidity_sweep.get('type') == 'BELOW_LOW':
                    bullish_conditions += 1
                if fvg.get('type') == 'BULLISH':
                    bullish_conditions += 1
                if order_block.get('type') == 'BULLISH':
                    bullish_conditions += 1
                if dealing_range['zone'] == 'DISCOUNT':
                    bullish_conditions += 0.5
                if ote_level['valid']:
                    bullish_conditions += 0.5
                
                if bullish_conditions >= 2.0:
                    confidence = self._calculate_confluence_score(signal_strengths)
                    
                    if liquidity_sweep['detected']:
                        stop_loss = liquidity_sweep['swept_level'] - (current['close'] * 0.001)
                    else:
                        stop_loss = current['close'] * 0.98
                    
                    return {
                        'action': 'BUY',
                        'bias': 'BULLISH',
                        'confidence': confidence,
                        'reasoning': f'ICT Bullish: {", ".join(signals_present)} (Score: {bullish_conditions:.1f})',
                        'entry_price': current['close'],
                        'stop_loss': stop_loss,
                        'take_profit': self._find_next_liquidity_pool(data, index, 'UP')
                    }
                return {'action': 'HOLD', 'bias': 'BULLISH', 'confidence': 0.0, 'reasoning': 'ICT conditions not aligned'}
            
            # BEARISH SETUP - Need at least 2 of 3 main signals
            elif market_bias == 'BEARISH':
                bearish_conditions = 0
                
                if liquidity_sweep.get('type') == 'ABOVE_HIGH':
                    bearish_conditions += 1
                if fvg.get('type') == 'BEARISH':
                    bearish_conditions += 1
                if order_block.get('type') == 'BEARISH':
                    bearish_conditions += 1
                if dealing_range['zone'] == 'PREMIUM':
                    bearish_conditions += 0.5
                if ote_level['valid']:
                    bearish_conditions += 0.5
                
                if bearish_conditions >= 2.0:
                    confidence = self._calculate_confluence_score(signal_strengths)
                    
                    if liquidity_sweep['detected']:
                        stop_loss = liquidity_sweep['swept_level'] + (current['close'] * 0.001)
                    else:
                        stop_loss = current['close'] * 1.02
                    
                    return {
                        'action': 'SELL',
                        'bias': 'BEARISH',
                        'confidence': confidence,
                        'reasoning': f'ICT Bearish: {", ".join(signals_present)} (Score: {bearish_conditions:.1f})',
                        'entry_price': current['close'],
                        'stop_loss': stop_loss,
                        'take_profit': self._find_next_liquidity_pool(data, index, 'DOWN')
                    }
                return {'action': 'HOLD', 'bias': 'BEARISH', 'confidence': 0.0, 'reasoning': 'ICT conditions not aligned'}
            
            return {'action': 'HOLD', 'bias': 'NEUTRAL', 'confidence': 0.0, 'reasoning': 'ICT conditions not fully aligned'}
                
        except Exception as e:
            logger.error(f"ICT AI analysis error: {e}")
            return {'action': 'HOLD', 'bias': 'NEUTRAL', 'confidence': 0.0, 'reasoning': 'Analysis failed'}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # Moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def _determine_market_bias(self, df: pd.DataFrame, index: int) -> str:
        """Determine market bias using MSS (Market Structure Shift)"""
        try:
            lookback = 50  # More lookback for M5 timeframe
            if index < lookback:
                return 'NEUTRAL'
            
            recent_data = df.iloc[index-lookback:index+1]
            highs = recent_data['high'].rolling(3, center=True).max()  # Smaller window for M5
            lows = recent_data['low'].rolling(3, center=True).min()
            current_price = df.iloc[index]['close']
            
            recent_high = highs.max()
            recent_low = lows.min()
            
            if current_price > recent_high * 0.9995:  # More sensitive for M5
                return 'BULLISH'
            elif current_price < recent_low * 1.0005:  # More sensitive for M5
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'
    
    def _detect_liquidity_sweep(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect liquidity sweeps below lows or above highs"""
        try:
            lookback = 20  # More lookback for M5 timeframe
            if index < lookback:
                return {'detected': False}
            
            current = df.iloc[index]
            recent_data = df.iloc[index-lookback:index]
            swing_low = recent_data['low'].min()
            swing_high = recent_data['high'].max()
            
            if current['low'] < swing_low and current['close'] > swing_low:
                return {'detected': True, 'type': 'BELOW_LOW', 'swept_level': swing_low, 'strength': 0.8}
            elif current['high'] > swing_high and current['close'] < swing_high:
                return {'detected': True, 'type': 'ABOVE_HIGH', 'swept_level': swing_high, 'strength': 0.8}
            
            return {'detected': False}
        except Exception:
            return {'detected': False}
    
    def _detect_fair_value_gap(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect Fair Value Gaps (FVG)"""
        try:
            if index < 3:
                return {'detected': False}
            
            bar1 = df.iloc[index-2]
            bar2 = df.iloc[index-1]
            bar3 = df.iloc[index]
            
            if bar1['high'] < bar3['low']:
                gap_size = bar3['low'] - bar1['high']
                if gap_size > (bar2['close'] * 0.0005):
                    return {'detected': True, 'type': 'BULLISH', 'high': bar3['low'], 'low': bar1['high'], 'strength': min(gap_size / (bar2['close'] * 0.002), 1.0)}
            elif bar1['low'] > bar3['high']:
                gap_size = bar1['low'] - bar3['high']
                if gap_size > (bar2['close'] * 0.0005):
                    return {'detected': True, 'type': 'BEARISH', 'high': bar1['low'], 'low': bar3['high'], 'strength': min(gap_size / (bar2['close'] * 0.002), 1.0)}
            
            return {'detected': False}
        except Exception:
            return {'detected': False}
    
    def _analyze_dealing_range(self, df: pd.DataFrame, index: int) -> Dict:
        """Analyze if price is in discount or premium zone"""
        try:
            lookback = 20
            if index < lookback:
                return {'zone': 'NEUTRAL'}
            
            recent_data = df.iloc[index-lookback:index+1]
            range_high = recent_data['high'].max()
            range_low = recent_data['low'].min()
            current_price = df.iloc[index]['close']
            range_50 = range_low + (range_high - range_low) * 0.5
            
            if current_price < range_50:
                return {'zone': 'DISCOUNT', 'level': range_50}
            else:
                return {'zone': 'PREMIUM', 'level': range_50}
        except Exception:
            return {'zone': 'NEUTRAL'}
    
    def _detect_order_block(self, df: pd.DataFrame, index: int, bias: str) -> Dict:
        """Detect institutional order blocks"""
        try:
            lookback = 15
            if index < lookback:
                return {'detected': False}
            
            for i in range(index-lookback, index-1):
                bar = df.iloc[i]
                next_bar = df.iloc[i+1]
                price_change = abs(next_bar['close'] - bar['close']) / bar['close']
                
                if bias == 'BULLISH' and price_change > 0.002 and next_bar['close'] > bar['close']:  # 0.2% for M5
                    return {'detected': True, 'type': 'BULLISH', 'high': bar['high'], 'low': bar['low'], 'strength': min(price_change * 10, 1.0)}
                elif bias == 'BEARISH' and price_change > 0.002 and next_bar['close'] < bar['close']:  # 0.2% for M5
                    return {'detected': True, 'type': 'BEARISH', 'high': bar['high'], 'low': bar['low'], 'strength': min(price_change * 10, 1.0)}
            
            return {'detected': False}
        except Exception:
            return {'detected': False}
    
    def _check_ote_levels(self, df: pd.DataFrame, index: int, fvg: Dict) -> Dict:
        """Check Optimal Trade Entry levels (62%-79% Fibonacci)"""
        try:
            if not fvg.get('detected'):
                return {'valid': False}
            
            current_price = df.iloc[index]['close']
            fvg_range = fvg['high'] - fvg['low']
            ote_62 = fvg['low'] + (fvg_range * 0.62)
            ote_79 = fvg['low'] + (fvg_range * 0.79)
            
            if ote_62 <= current_price <= ote_79:
                return {'valid': True, 'strength': 0.9, 'level_62': ote_62, 'level_79': ote_79}
            
            return {'valid': False}
        except Exception:
            return {'valid': False}
    
    def _find_next_liquidity_pool(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """Find next liquidity pool for take profit"""
        try:
            lookback = 30
            current_price = df.iloc[index]['close']
            
            if direction == 'UP':
                recent_highs = df.iloc[max(0, index-lookback):index]['high']
                resistance = recent_highs[recent_highs > current_price].min()
                return resistance if not pd.isna(resistance) else current_price * 1.02
            else:
                recent_lows = df.iloc[max(0, index-lookback):index]['low']
                support = recent_lows[recent_lows < current_price].max()
                return support if not pd.isna(support) else current_price * 0.98
        except Exception:
            return current_price * (1.02 if direction == 'UP' else 0.98)
    
    def _calculate_confluence_score(self, strengths: List[float]) -> float:
        """Calculate confluence score from multiple signal strengths"""
        try:
            if not strengths:
                return 0.0
            base_score = sum(strengths) / len(strengths)
            confluence_bonus = min(len(strengths) * 0.1, 0.3)
            return min(base_score + confluence_bonus, 1.0)
        except Exception:
            return 0.0

    def remember_trade(self, trade_data: Dict):
        """Store trade in memory for learning"""
        self.memories.append({
            'timestamp': datetime.now(),
            'symbol': trade_data.get('symbol'),
            'action': trade_data.get('action'),
            'success': trade_data.get('pnl', 0) > 0,
            'pnl': trade_data.get('pnl', 0)
        })
        
        # Keep only last 1000 memories
        if len(self.memories) > 1000:
            self.memories = self.memories[-1000:]

class ICTAnalyzer:
    """ICT/SMC Strategy Implementation"""
    
    def __init__(self):
        self.lookback_periods = 20
        
    def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure for BOS/ChoCH"""
        try:
            highs = data['high'].rolling(self.lookback_periods).max()
            lows = data['low'].rolling(self.lookback_periods).min()
            
            current_high = data['high'].iloc[-1]
            current_low = data['low'].iloc[-1]
            prev_high = highs.iloc[-2] if len(highs) > 1 else current_high
            prev_low = lows.iloc[-2] if len(lows) > 1 else current_low
            
            # Simple structure analysis
            structure = "NEUTRAL"
            if current_high > prev_high:
                structure = "BULLISH_BOS"
            elif current_low < prev_low:
                structure = "BEARISH_BOS"
                
            return {
                'structure': structure,
                'strength': 0.6,
                'key_levels': {
                    'resistance': float(highs.iloc[-1]),
                    'support': float(lows.iloc[-1])
                }
            }
        except Exception as e:
            logger.error(f"Structure analysis error: {e}")
            return {'structure': 'NEUTRAL', 'strength': 0.0, 'key_levels': {}}
    
    def detect_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """Detect institutional order blocks"""
        try:
            order_blocks = []
            
            # Simple order block detection
            for i in range(10, len(data) - 5):
                high = data['high'].iloc[i]
                low = data['low'].iloc[i]
                volume = data.get('tick_volume', pd.Series([1] * len(data))).iloc[i]
                
                # Check for significant price movement
                prev_close = data['close'].iloc[i-1]
                curr_close = data['close'].iloc[i]
                price_change = abs(curr_close - prev_close) / prev_close
                
                if price_change > 0.002 and volume > data.get('tick_volume', pd.Series([1] * len(data))).rolling(20).mean().iloc[i]:
                    order_blocks.append({
                        'type': 'BULLISH' if curr_close > prev_close else 'BEARISH',
                        'high': float(high),
                        'low': float(low),
                        'strength': min(price_change * 100, 1.0),
                        'timestamp': data.index[i] if hasattr(data.index[i], 'strftime') else datetime.now()
                    })
            
            return order_blocks[-5:]  # Return last 5 order blocks
        except Exception as e:
            logger.error(f"Order block detection error: {e}")
            return []

class MT5Broker:
    """MetaTrader 5 Broker Interface"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to MT5 with robust retries and session clearing"""
        try:
            # Force close any hung sessions first
            mt5.shutdown()
            await asyncio.sleep(1)
            
            terminal_path = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
            
            success = False
            for i in range(3):
                logger.info(f"Connecting to MT5 (Attempt {i+1}/3)...")
                if mt5.initialize(path=terminal_path):
                    success = True
                    break
                logger.warning(f"Connection attempt {i+1} failed: {mt5.last_error()}")
                await asyncio.sleep(2)
                
            if not success:
                logger.error(f"MT5 could not be initialized after 3 attempts: {mt5.last_error()}")
                return False
                
            # Login with credentials
            login = self.config.get('login') or int(os.getenv('MT5_LOGIN', 0))
            password = self.config.get('password') or os.getenv('MT5_PASSWORD')
            server = self.config.get('server') or os.getenv('MT5_SERVER')
            
            logger.info(f"Logging into {server} (Account: {login})...")
            if login and password and server:
                if not mt5.login(login, password=password, server=server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    # Fail-safe: Check if we are already logged in to this account
                    acc = mt5.account_info()
                    if acc and acc.login == login:
                        logger.info("Terminal is already logged into the correct account manually. Proceeding.")
                    else:
                        return False
            
            self.connected = True
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Connected to MT5 - Balance: ${account_info.balance:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """Check if MT5 is still connected and responsive"""
        try:
            # Check account info as a heartbeat
            acc = mt5.account_info()
            if acc is None:
                self.connected = False
                return False
            # Check terminal connection status
            terminal = mt5.terminal_info()
            if terminal and not terminal.connected:
                self.connected = False
                return False
            self.connected = True
            return True
        except:
            self.connected = False
            return False

    def get_market_data(self, symbol: str, timeframe: str = "M5", count: int = 500) -> pd.DataFrame:
        """Get market data from MT5"""
        try:
            # Map timeframe
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            timeframe_mt5 = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    async def place_pending_order(self, symbol: str, order_type: int, volume: float, price: float, magic: int) -> Dict:
        """Place a pending limit order"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
            
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "magic": magic,
                "comment": "GRID_ENTRY",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN, # Changed to RETURN for better compatibility
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {'success': False, 'error': f'Code {result.retcode}: {result.comment}'}
            
            return {'success': True, 'ticket': result.order}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def cancel_all_pendings(self, symbol: str):
        """Cancel all pending orders for a specific symbol"""
        try:
            orders = mt5.orders_get(symbol=symbol)
            if orders is None: return
            
            for o in orders:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": o.ticket
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to cancel order {o.ticket}: {result.retcode}")
            logger.info(f"🧹 Cleaned all existing pending orders for {symbol}")
        except Exception as e:
            logger.error(f"Error canceling pendings: {e}")

    def close_all_side(self, symbol: str, side: str, magic: int = None):
        """Close all positions for a specific side (BUY/SELL)"""
        try:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return
            
            for pos in positions:
                pos_side = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
                if pos_side == side and (magic is None or pos.magic == magic):
                    # Close position
                    action = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": action,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": pos.magic,
                        "comment": "CLOSE_GRID",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(request)
        except Exception as e:
            logger.error(f"Error closing side {side}: {e}")

    def place_order(self, symbol: str, action: str, volume: float, price: float, 
                   stop_loss: float = None, take_profit: float = None) -> Dict:
        """Place trading order"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "NEXT_LEVEL_BRAIN",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {'success': False, 'error': f'Order failed: {result.retcode}'}
            
            logger.info(f"Order placed: {action} {volume} {symbol} at {price}")
            return {
                'success': True,
                'ticket': result.order,
                'price': result.price,
                'volume': result.volume
            }
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return []
                
            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'magic': pos.magic,
                    'time': datetime.fromtimestamp(pos.time)
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

class RiskManager:
    """Risk Management System"""
    
    def __init__(self, config: Dict):
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.daily_pnl = 0.0
        
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_loss: float, symbol: str) -> float:
        """Calculate optimal position size"""
        try:
            # Risk amount
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Price difference
            price_diff = abs(entry_price - stop_loss)
            if price_diff == 0:
                return 0.01  # Minimum position size
            
            # Calculate position size based on asset type
            if 'BTC' in symbol or 'ETH' in symbol:
                # For crypto, use smaller position sizes
                position_size = min(risk_amount / (price_diff * 10), 0.1)
            elif 'XAU' in symbol or 'XAG' in symbol:
                # For metals
                position_size = min(risk_amount / price_diff, 1.0)
            else:
                # For forex
                position_size = min(risk_amount / (price_diff * 100000), 1.0)
            
            # Round to valid lot size
            position_size = round(position_size, 2)
            
            # Ensure minimum and maximum limits
            return max(0.01, min(position_size, 1.0))
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.01
    
    def check_risk_limits(self, account_balance: float, current_drawdown: float) -> bool:
        """Check if trading is allowed based on risk limits"""
        # Check daily loss limit
        if abs(self.daily_pnl) > account_balance * self.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False
            
        # Check maximum drawdown
        if current_drawdown > self.max_drawdown:
            logger.warning("Maximum drawdown limit reached")
            return False
            
        return True

class GridManager:
    """Manages User Requested Grid Strategy"""
    def __init__(self, broker, config: Dict = None):
        self.broker = broker
        grid_config = (config or {}).get('grid', {})
        
        self.magic_buy = 777001
        self.magic_sell = 777002
        self.grid_size = grid_config.get('size', 300)
        self.spacing = grid_config.get('spacing', 1.0)  # Spacing in $
        self.lot_size = grid_config.get('lot_size', 0.01)
        self.profit_threshold_pct = grid_config.get('profit_target_pct', 0.25) # 25% profit target
        self.profit_target_usd = grid_config.get('profit_target_usd', 0) # Fixed $ profit target
        self.mode = grid_config.get('mode', 'BOTH') # Added mode: BOTH, BUY_ONLY, SELL_ONLY
        self.active_grids = {} # symbol -> {'type': 'BUY/SELL', 'base_price': float, 'last_index': int}
        self.batch_size = 50
        self.trigger_threshold = 10 # Place next batch if only 10 pendings left (meaning 40 hit)
        self.total_target = self.grid_size # Default 300
        
        # State Persistence
        self.state_file = Path("logs/grid_state.json")
        self._load_state()

    def _save_state(self):
        """Save grid progress to file"""
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.active_grids, f)
        except Exception as e:
            logger.error(f"Failed to save grid state: {e}")

    def _load_state(self):
        """Load grid progress from file"""
        try:
            if self.state_file.exists() and self.state_file.stat().st_size > 0:
                with open(self.state_file, 'r') as f:
                    self.active_grids = json.load(f)
                logger.info(f"📁 Loaded Grid State from {self.state_file}")
            else:
                self.active_grids = {}
        except json.JSONDecodeError:
            logger.warning("⚠️ Grid state file was corrupt. Starting fresh.")
            self.active_grids = {}
        except Exception as e:
            logger.error(f"Failed to load grid state: {e}")
            self.active_grids = {}

    async def update(self, symbol, current_price, bias, balance):
        """Update grid logic based on bias and profit"""
        try:
            # 1. Check profit targets for existing positions
            positions = self.broker.get_positions()
            symbol_positions = [p for p in positions if p['symbol'] == symbol]
            
            # Filter Grid positions
            buy_positions = [p for p in symbol_positions if p['type'] == 'BUY' and p.get('magic') == self.magic_buy]
            sell_positions = [p for p in symbol_positions if p['type'] == 'SELL' and p.get('magic') == self.magic_sell]
            
            buy_profit = sum(p['profit'] for p in buy_positions)
            sell_profit = sum(p['profit'] for p in sell_positions)
            
            # 1. Total Grid Profit Check (Combined BUY + SELL)
            total_grid_profit = buy_profit + sell_profit
            total_grid_lots = sum(p['volume'] for p in buy_positions) + sum(p['volume'] for p in sell_positions)
            
            if total_grid_lots > 0:
                # Target: $0.50 profit per 0.01 lot lot
                total_target = (total_grid_lots / 0.01) * 0.50
                
                # Use current time for periodic logging
                now = time.time()
                if not hasattr(self, '_last_log_time'): self._last_log_time = 0
                if now - self._last_log_time > 5:
                    logger.info(f"📊 Basket Status: ${total_grid_profit:.2f} / Target: ${total_target:.2f}")
                    self._last_log_time = now

                if total_grid_profit >= total_target:
                    logger.info(f"🚀 PROFITS DETECTED: ${total_grid_profit:.2f}. Closing all active grid trades!")
                    # Close everything for this symbol
                    self.broker.close_all_side(symbol, 'BUY', self.magic_buy)
                    self.broker.close_all_side(symbol, 'SELL', self.magic_sell)
                    self.broker.cancel_all_pendings(symbol)
                    self.active_grids.clear()
                    self._save_state()
                    # Removed return to allow immediate re-injection in same cycle

            # Individual side checks (Optional backup)
            if buy_positions:
                target_buy = (sum(p['volume'] for p in buy_positions) / 0.01) * 0.50
                if buy_profit >= target_buy:
                    logger.info(f"🎯 BUY Basket Target Hit (${buy_profit:.2f}). Closing BUY side...")
                    self.broker.close_all_side(symbol, 'BUY', self.magic_buy)
                    self.broker.cancel_all_pendings(symbol)
                    if symbol in self.active_grids and self.active_grids[symbol]['type'] == 'BUY':
                        del self.active_grids[symbol]
                        self._save_state()

            if sell_positions:
                target_sell = (sum(p['volume'] for p in sell_positions) / 0.01) * 0.50
                if sell_profit >= target_sell:
                    logger.info(f"🎯 SELL Basket Target Hit (${sell_profit:.2f}). Closing SELL side...")
                    self.broker.close_all_side(symbol, 'SELL', self.magic_sell)
                    self.broker.cancel_all_pendings(symbol)
                    if symbol in self.active_grids and self.active_grids[symbol]['type'] == 'SELL':
                        del self.active_grids[symbol]
                        self._save_state()

            # 2. Place Grid
            # Check Account Limits (Fix for Error 10033)
            acc_info = mt5.account_info()
            effective_grid_size = self.grid_size
            if acc_info and acc_info.limit_orders > 0:
                # If broker limit is e.g. 100, we must stay below it
                # We subtract current positions/pendings to be safe
                active_count = len(positions) + (len(mt5.orders_get()) if mt5.orders_get() else 0)
                available = max(0, acc_info.limit_orders - active_count - 5)
                if effective_grid_size > available:
                    effective_grid_size = available
                    if available < 5: 
                        return # Not enough room for a grid

            # SELL Grid Placement (Bullish Bias OR SELL_ONLY mode)
            if (self.mode == 'SELL_ONLY' or (self.mode == 'BOTH' and bias == 'BULLISH')):
                active_pendings = mt5.orders_get(symbol=symbol)
                sell_pendings = [o for o in active_pendings if o.magic == self.magic_sell] if active_pendings else []
                sell_positions = [p for p in symbol_positions if p['type'] == 'SELL' and p.get('magic') == self.magic_sell]
                
                # Check if we need to start or add a batch
                grid_info = self.active_grids.get(symbol, {})
                if grid_info.get('type') != 'SELL': grid_info = {} # Reset if direction changed
                
                last_index = grid_info.get('last_index', 0)
                
                # Trigger: No pendings/positions OR only trigger_threshold pendings left AND haven't reached total_target
                if (not sell_pendings and not sell_positions) or (len(sell_pendings) <= self.trigger_threshold and last_index < self.total_target):
                    
                    if not sell_pendings and not sell_positions:
                        last_index = 0 # Fresh start
                        base_price = current_price
                        logger.info(f"🕸️ Starting NEW SELL Grid for {symbol} at {base_price}")
                    else:
                        base_price = grid_info.get('base_price', current_price)
                        logger.info(f"🔄 Staggered Batch: Placing next set of SELL orders for {symbol} (Last Index: {last_index})")

                    num_to_place = min(self.batch_size, self.total_target - last_index)
                    if num_to_place > 0:
                        success_count = 0
                        for i in range(1, num_to_place + 1):
                            k = last_index + i
                            entry_price = base_price + (k * self.spacing)
                            res = await self.broker.place_pending_order(symbol, mt5.ORDER_TYPE_SELL_LIMIT, self.lot_size, entry_price, self.magic_sell)
                            if res['success']: success_count += 1
                        
                        if success_count > 0:
                            self.active_grids[symbol] = {
                                'type': 'SELL', 
                                'base_price': base_price,
                                'last_index': last_index + success_count
                            }
                            self._save_state()
                            logger.info(f"✅ {success_count} SELL orders added. Total placed in sequence: {last_index + success_count}/{self.total_target}")

                # Case A: Market pushes HIGHER (Trend Extension) - Increase grid depth
                # Ensure base_price is available for logic below
                bp = grid_info.get('base_price', current_price) if 'base_price' not in locals() else base_price
                
                if current_price > bp + self.spacing:
                    new_dist = current_price - bp
                    new_orders_needed = int(new_dist / self.spacing)
                    
                    # Only expand if the market has pushed FURTHER than our already placed orders
                    if new_orders_needed > last_index:
                        num_to_add = min(new_orders_needed - last_index, self.total_target - last_index)
                        if num_to_add > 0:
                            logger.info(f"📈 Market pushed higher ({current_price}). Expanding SELL grid by {num_to_add} orders.")
                            success_count = 0
                            for i in range(1, num_to_add + 1):
                                k = last_index + i
                                entry_price = bp + (k * self.spacing)
                                res = await self.broker.place_pending_order(symbol, mt5.ORDER_TYPE_SELL_LIMIT, self.lot_size, entry_price, self.magic_sell)
                                if res['success']: success_count += 1
                            
                            if success_count > 0:
                                self.active_grids[symbol]['last_index'] += success_count
                                self._save_state()

                # Case B: Market pushes LOWER (Gap Filling) - No positions open, shift grid to follow market down
                elif current_price < bp - self.spacing and not sell_positions:
                    jump = int((bp - current_price) / self.spacing)
                    num_to_add = min(jump, self.batch_size, self.total_target - last_index)
                    if num_to_add > 0:
                        logger.info(f"📉 Market moved down ({current_price}). Shifting SELL grid DOWN by {num_to_add} levels.")
                        new_base = bp - (num_to_add * self.spacing)
                        success_count = 0
                        for k in range(1, num_to_add + 1):
                            entry_price = new_base + (k * self.spacing)
                            res = await self.broker.place_pending_order(symbol, mt5.ORDER_TYPE_SELL_LIMIT, self.lot_size, entry_price, self.magic_sell)
                            if res['success']: success_count += 1
                        
                        if success_count > 0:
                            self.active_grids[symbol]['base_price'] = bp - (success_count * self.spacing)
                            self.active_grids[symbol]['last_index'] += success_count
                            self._save_state()

            # BUY Grid Placement (Bearish Bias OR BUY_ONLY mode)
            if (self.mode == 'BUY_ONLY' or (self.mode == 'BOTH' and bias == 'BEARISH')):
                active_pendings = mt5.orders_get(symbol=symbol)
                buy_pendings = [o for o in active_pendings if o.magic == self.magic_buy] if active_pendings else []
                buy_positions = [p for p in symbol_positions if p['type'] == 'BUY' and p.get('magic') == self.magic_buy]

                # Check if we need to start or add a batch
                grid_info = self.active_grids.get(symbol, {})
                if grid_info.get('type') != 'BUY': grid_info = {} # Reset if direction changed

                last_index = grid_info.get('last_index', 0)

                # Trigger: No pendings/positions OR only trigger_threshold pendings left AND haven't reached total_target
                if (not buy_pendings and not buy_positions) or (len(buy_pendings) <= self.trigger_threshold and last_index < self.total_target):
                    
                    if not buy_pendings and not buy_positions:
                        last_index = 0 # Fresh start
                        base_price = current_price
                        logger.info(f"🕸️ Starting NEW BUY Grid for {symbol} at {base_price}")
                    else:
                        base_price = grid_info.get('base_price', current_price)
                        logger.info(f"🔄 Staggered Batch: Placing next set of BUY orders for {symbol} (Last Index: {last_index})")

                    num_to_place = min(self.batch_size, self.total_target - last_index)
                    if num_to_place > 0:
                        success_count = 0
                        for i in range(1, num_to_place + 1):
                            k = last_index + i
                            entry_price = base_price - (k * self.spacing)
                            res = await self.broker.place_pending_order(symbol, mt5.ORDER_TYPE_BUY_LIMIT, self.lot_size, entry_price, self.magic_buy)
                            if res['success']: success_count += 1
                        
                        if success_count > 0:
                            self.active_grids[symbol] = {
                                'type': 'BUY', 
                                'base_price': base_price,
                                'last_index': last_index + success_count
                            }
                            self._save_state()
                            logger.info(f"✅ {success_count} BUY orders added. Total placed in sequence: {last_index + success_count}/{self.total_target}")

                # Case A: Market pushes LOWER (Trend Extension) - Increase grid depth
                # Ensure base_price is available for logic below
                bp = grid_info.get('base_price', current_price) if 'base_price' not in locals() else base_price

                if current_price < bp - self.spacing:
                    new_dist = bp - current_price
                    new_orders_needed = int(new_dist / self.spacing)
                    
                    # Only expand if the market has pushed FURTHER than our already placed orders
                    if new_orders_needed > last_index:
                        num_to_add = min(new_orders_needed - last_index, self.total_target - last_index)
                        if num_to_add > 0:
                            logger.info(f"📉 Market pushed lower ({current_price}). Expanding BUY grid by {num_to_add} orders.")
                            success_count = 0
                            for i in range(1, num_to_add + 1):
                                k = last_index + i
                                entry_price = bp - (k * self.spacing)
                                res = await self.broker.place_pending_order(symbol, mt5.ORDER_TYPE_BUY_LIMIT, self.lot_size, entry_price, self.magic_buy)
                                if res['success']: success_count += 1
                            
                            if success_count > 0:
                                self.active_grids[symbol]['last_index'] += success_count
                                self._save_state()

                # Case B: Market pushes HIGHER (Gap Filling) - No positions open, shift grid to follow market up
                elif current_price > bp + self.spacing and not buy_positions:
                    jump = int((current_price - bp) / self.spacing)
                    num_to_add = min(jump, self.batch_size, self.total_target - last_index)
                    if num_to_add > 0:
                        logger.info(f"📈 Market moved up ({current_price}). Shifting BUY grid UP by {num_to_add} levels.")
                        new_base = bp + (num_to_add * self.spacing)
                        success_count = 0
                        for k in range(1, num_to_add + 1):
                            entry_price = new_base - (k * self.spacing)
                            res = await self.broker.place_pending_order(symbol, mt5.ORDER_TYPE_BUY_LIMIT, self.lot_size, entry_price, self.magic_buy)
                            if res['success']: success_count += 1
                        
                        if success_count > 0:
                            self.active_grids[symbol]['base_price'] = bp + (success_count * self.spacing)
                            self.active_grids[symbol]['last_index'] += success_count
                            self._save_state()

        except Exception as e:
            logger.error(f"Error in grid update: {e}")

class LiveTradingSystem:
    """Main Live Trading System"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.broker = MT5Broker(self.config.get('mt5', {}))
        self.ai_brain = TradingBrain()
        self.ict_analyzer = ICTAnalyzer()
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        self.grid_manager = GridManager(self.broker, self.config)
        
        self.running = False
        self.symbols = self.config.get('symbols', ['XAUUSDm'])
        self.timeframe = self.config.get('timeframe', 'M5')
        self.strategy = "ICT SMC" # Default strategy
        
        # Performance tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.start_balance = 0.0
        self.trade_history = []
        self.reports_dir = Path("logs/live_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default configuration
                return {
                    'mt5': {
                        'login': None,
                        'password': None,
                        'server': None
                    },
                    'symbols': ['EURUSDm', 'GBPUSDm', 'XAUUSDm'],
                    'timeframe': 'M5',
                    'risk': {
                        'max_risk_per_trade': 0.02,
                        'max_daily_loss': 0.05,
                        'max_drawdown': 0.15
                    }
                }
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """Initialize the trading system"""
        try:
            logger.info("🧠 Initializing NEXT LEVEL BRAIN Live Trading System...")
            
            # Connect to broker
            if not await self.broker.connect():
                logger.error("Failed to connect to MT5")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                self.start_balance = account_info.balance
                logger.info(f"Account Balance: ${self.start_balance:.2f}")
            
            # Cleanup old pending orders first
            for symbol in self.symbols:
                self.broker.cancel_all_pendings(symbol)
            
            logger.info("✅ System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    async def analyze_and_trade(self, symbol: str):
        """Analyze market and execute trades for a symbol"""
        try:
            # Get account context
            acc = mt5.account_info()
            if not acc: return
            balance = acc.balance

            # Get market data
            data = self.broker.get_market_data(symbol, self.timeframe, 500)
            if data.empty:
                return
            
            # AI Analysis (Keep for bias detection)
            ai_analysis = self.ai_brain.analyze_market(symbol, data)
            current_price = data['close'].iloc[-1]
            bias = ai_analysis.get('bias', 'NEUTRAL')
            
            # 1. Grid Strategy Logic
            if "Grid" in self.strategy:
                # Set mode based on strategy name
                if "BUY ONLY" in self.strategy:
                    self.grid_manager.mode = "BUY_ONLY"
                elif "SELL ONLY" in self.strategy:
                    self.grid_manager.mode = "SELL_ONLY"
                else:
                    self.grid_manager.mode = "BOTH"
                
                await self.grid_manager.update(symbol, current_price, bias, balance)
            
            # 2. ICT SMC Strategy Logic
            elif self.strategy == "ICT SMC":
                if ai_analysis['action'] in ['BUY', 'SELL'] and ai_analysis['confidence'] >= 0.70:
                    logger.info(f"🎯 ICT Signal Detected: {ai_analysis['action']} {symbol} (Confidence: {ai_analysis['confidence']:.2f})")
                    await self._execute_trade(symbol, ai_analysis, {})
                
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
    
    async def _execute_trade(self, symbol: str, ai_analysis: Dict, structure: Dict):
        """Execute a trade based on analysis"""
        try:
            # Get account balance
            account_info = mt5.account_info()
            if not account_info:
                return
            
            balance = account_info.balance
            
            # Check risk limits
            current_drawdown = max(0, (self.start_balance - balance) / self.start_balance) if self.start_balance else 0
            if not self.risk_manager.check_risk_limits(balance, current_drawdown):
                logger.warning(f"Risk limits prevent trading {symbol}")
                return
            
            # Calculate position size
            entry_price = ai_analysis['entry_price']
            stop_loss = ai_analysis['stop_loss']
            take_profit = ai_analysis['take_profit']
            
            position_size = self.risk_manager.calculate_position_size(
                balance, entry_price, stop_loss, symbol
            )
            
            # Place order
            result = self.broker.place_order(
                symbol=symbol,
                action=ai_analysis['action'],
                volume=position_size,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result['success']:
                self.trades_today += 1
                logger.info(f"✅ Trade executed: {ai_analysis['action']} {symbol}")
                logger.info(f"📊 Reasoning: {ai_analysis['reasoning']}")
                
                # Remember trade for AI learning
                self.ai_brain.remember_trade({
                    'symbol': symbol,
                    'action': ai_analysis['action'],
                    'entry_price': entry_price,
                    'confidence': ai_analysis['confidence']
                })
            else:
                logger.error(f"❌ Trade failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            positions = self.broker.get_positions()
            
            for pos in positions:
                # Simple trailing stop logic
                if pos['profit'] > 50:  # If profit > $50
                    logger.info(f"💰 Position {pos['symbol']} in profit: ${pos['profit']:.2f}")
                elif pos['profit'] < -100:  # If loss > $100
                    logger.warning(f"⚠️ Position {pos['symbol']} in loss: ${pos['profit']:.2f}")
                    
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
    
    def display_status(self):
        """Display current system status"""
        try:
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance
                daily_change = current_balance - self.start_balance
                
                print(f"\n{'='*50}")
                print(f"🧠 NEXT LEVEL BRAIN - LIVE TRADING STATUS")
                print(f"{'='*50}")
                print(f"💰 Balance: ${current_balance:.2f}")
                print(f"📈 Daily P&L: ${daily_change:.2f}")
                print(f"📊 Trades Today: {self.trades_today}")
                print(f"🎯 AI Memories: {len(self.ai_brain.memories)}")
                print(f"⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}")
                print(f"📂 Reports Path: {self.reports_dir.absolute()}")
                
                positions = self.broker.get_positions()
                if positions:
                    print(f"📋 Open Positions: {len(positions)}")
                    for pos in positions:
                        print(f"  {pos['symbol']}: {pos['type']} ${pos['profit']:.2f}")
                else:
                    print(f"📋 Open Positions: 0")
                print(f"{'='*50}")
                
        except Exception as e:
            logger.error(f"Status display error: {e}")
    
    async def run(self):
        """Main trading loop with Auto-Reconnect Watchdog"""
        try:
            if not await self.initialize():
                return
            
            self.running = True
            logger.info("🚀 Starting live trading...")
            
            cycle_count = 0
            
            while self.running:
                try:
                    # Connection Watchdog
                    if not await self.broker.is_connected():
                        logger.warning("📡 Connection lost! Attempting to reconnect...")
                        # Wait exponentially or simply retry
                        if await self.broker.connect():
                            logger.info("✅ Reconnected successfully!")
                        else:
                            logger.error("❌ Reconnect failed. Retrying in 10 seconds...")
                            await asyncio.sleep(10)
                            continue

                    cycle_count += 1
                    
                    # Display status every 10 cycles
                    if cycle_count % 10 == 0:
                        self.display_status()
                    
                    # Analyze each symbol
                    for symbol in self.symbols:
                        await self.analyze_and_trade(symbol)
                        await asyncio.sleep(1)  # Small delay between symbols
                    
                    # Monitor positions
                    await self.monitor_positions()
                    
                    # Update session history and save partial report
                    self._update_session_history()
                    if cycle_count % 20 == 0:
                        self.generate_session_report()
                    
                    # Wait before next cycle
                    await asyncio.sleep(30)  # 30 second cycle
                    
                except KeyboardInterrupt:
                    logger.info("Shutdown signal received")
                    break
                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    # If it's a network error, MT5 calls might fail
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Fatal trading error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the trading system"""
        try:
            self.running = False
            logger.info("🛑 Shutting down trading system...")
            
            # Generate final report before closing
            self._update_session_history()
            report_file = self.generate_session_report()
            if report_file:
                logger.info(f"📄 Final Session Report saved: {report_file}")
            
            # Close MT5 connection
            mt5.shutdown()
            
            logger.info("✅ Shutdown completed")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    def _update_session_history(self):
        """Fetch closed deals from MT5 for the last 30 days to build performance metrics"""
        try:
            # Fetch last 30 days for comprehensive performance tracking
            from_date = datetime.now() - timedelta(days=30)
            to_date = datetime.now() + timedelta(days=1)
            
            deals = mt5.history_deals_get(from_date, to_date)
            if not deals:
                return

            new_history = []
            for d in deals:
                if d.entry == 1: # DEAL_ENTRY_OUT
                    new_history.append({
                        'ticket': d.ticket,
                        'symbol': d.symbol,
                        'type': 'BUY' if d.type == mt5.DEAL_TYPE_BUY else 'SELL',
                        'volume': d.volume,
                        'price': d.price,
                        'profit': d.profit + d.commission + d.swap,
                        'magic': d.magic,
                        'time': datetime.fromtimestamp(d.time).strftime('%Y-%m-%d %H:%M:%S'),
                        'comment': d.comment
                    })
            
            self.trade_history = new_history
            # Today's specific metrics for display
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_trades = [t for t in self.trade_history if today_str in t['time']]
            self.daily_pnl = sum(t['profit'] for t in today_trades)
            self.trades_today = len(today_trades)
            
        except Exception as e:
            logger.error(f"Failed to update history: {e}")

    def generate_session_report(self) -> Optional[str]:
        """Generate a detailed markdown report with Backtest-style metrics"""
        try:
            if not self.trade_history:
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.reports_dir / f"live_performance_{timestamp}.md"
            
            # Advanced metrics calculation
            profits = [t['profit'] for t in self.trade_history]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p <= 0]
            
            win_rate = (len(wins) / len(profits) * 100) if profits else 0
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
            
            # Drawdown calculation
            acc_info = mt5.account_info()
            base_balance = acc_info.balance if acc_info else 10000.0
            cum_pnl = np.cumsum(profits)
            equity_curve = base_balance + cum_pnl
            peak = np.maximum.accumulate(equity_curve)
            drawdown_pct = (peak - equity_curve) / peak * 100
            max_dd_pct = np.max(drawdown_pct) if len(drawdown_pct) > 0 else 0
            
            report = [
                "# 🧠 NEXT LEVEL BRAIN - LIVE PERFORMANCE REPORT",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Tracking Period:** Last 30 Days",
                f"**Strategy:** {self.strategy}",
                "\n## 📊 PERFORMANCE SUMMARY (Backtest Style)",
                f"- **Total Trades:** {len(profits)}",
                f"- **Win Rate:** {win_rate:.1f}%",
                f"- **Total P&L:** ${sum(profits):.2f}",
                f"- **Profit Factor:** {profit_factor:.2f}",
                f"- **Max Drawdown:** {max_dd_pct:.2f}%",
                f"- **Avg Win:** ${ (sum(wins)/len(wins)) if wins else 0 :.2f}",
                f"- **Avg Loss:** ${ (sum(losses)/len(losses)) if losses else 0 :.2f}",
                "\n## 📋 RECENT TRADE LOG (Last 50)",
                "| Time | Symbol | Side | Lots | Profit ($) | Comment |",
                "|------|--------|------|------|------------|---------|"
            ]
            
            for t in sorted(self.trade_history, key=lambda x: x['time'], reverse=True)[:50]:
                report.append(f"| {t['time']} | {t['symbol']} | {t['type']} | {t['volume']} | {t['profit']:.2f} | {t['comment']} |")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(report))
            
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None

def select_trade_setup():
    """Simplified setup selection as requested"""
    print("\n" + "="*60)
    print("🚀 NEXT LEVEL BRAIN - QUICK LAUNCH (GOLD ONLY)")
    print("="*60)
    
    # 1. Strategy/Direction
    print("\n🎯 SELECT DIRECTION / ACTION:")
    print("1. 📉 Grid BUY ONLY (300 Orders)")
    print("2. 📈 Grid SELL ONLY (300 Orders)")
    print("3. 🕸️ Grid BOTH (300+300 Orders)")
    print("4. 🧠 ICT SMC (Trend Following)")
    print("5. 📊 OPEN LIVE DASHBOARD (Visual Tracker)")
    print("6. 🧹 DELETE ALL PENDING ORDERS")
    
    strategy = "Grid Both"
    while True:
        choice = input("Choice (1-6): ").strip()
        if choice == "1": strategy = "Grid BUY ONLY"; break
        if choice == "2": strategy = "Grid SELL ONLY"; break
        if choice == "3": strategy = "Grid Both"; break
        if choice == "4": strategy = "ICT SMC"; break
        if choice == "5":
            print("🚀 Opening Live Dashboard...")
            import subprocess
            subprocess.Popen([sys.executable, "live_dashboard.py"], creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            print("✅ Dashboard launched. Continuing...")
            continue
        if choice == "6":
            print("🧹 Cleaning up all Gold pending orders...")
            if mt5.initialize():
                # Rapid cleanup
                for s in ["XAUUSDm", "XAUUSD"]:
                    orders = mt5.orders_get(symbol=s)
                    if orders:
                        for o in orders:
                            mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
                print("✅ All pending orders deleted.")
            else:
                print("❌ Failed to connect to MT5 for cleanup.")
            continue
        print("Invalid choice.")

    # 2. Timeframe
    print("\n⏰ SELECT TIMEFRAME:")
    tfs = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"]
    for i, tf in enumerate(tfs, 1):
        print(f"{i}. {tf}")
    
    timeframe = "M1"
    while True:
        try:
            choice = int(input(f"Choice (1-{len(tfs)}): ").strip())
            if 1 <= choice <= len(tfs):
                timeframe = tfs[choice-1]
                break
        except: pass
        print("Invalid choice.")

    return ["XAUUSDm"], strategy, timeframe


def main():
    """Main function - CLI mode"""
    try:
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("charts").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
        symbols, strategy, timeframe = select_trade_setup()
        if symbols is None or strategy is None or timeframe is None:
            print("Exiting.")
            return
        
        trading_system = LiveTradingSystem()
        trading_system.symbols = symbols
        trading_system.strategy = strategy
        trading_system.timeframe = timeframe
        
        # Run trading loop
        asyncio.run(trading_system.run())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
