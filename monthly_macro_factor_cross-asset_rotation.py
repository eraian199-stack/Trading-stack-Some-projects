# region imports
from AlgorithmImports import *

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
# endregion


class AIStocksBondsRotationAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(self.end_date - timedelta(5*365))
        self.settings.daily_precise_end_time = False
        self.settings.seed_initial_prices = True
        # Add securities
        self._bitcoin = self.add_crypto("BTCUSD", market=Market.BITFINEX, leverage=2).symbol
        self._equities = [self.add_equity(ticker).symbol for ticker in ['SPY', 'GLD', 'BND']]
        self._symbols = self._equities + [self._bitcoin]
        # Add FRED data
        self._factors = [
            self.add_data(Fred, ticker, Resolution.DAILY).symbol 
            for ticker in ['VIXCLS', 'T10Y3M', 'DFF']
        ]
        # ML model setup
        self._model = DecisionTreeRegressor(max_depth=12, random_state=1)
        self._scaler = StandardScaler()
        # Parameters
        self._gross_exposure = float(self.get_parameter("gross_exposure", 1.5))
        self._max_bitcoin_weight = float(self.get_parameter("max_bitcoin_weight", 0.1))
        lookback_years = int(self.get_parameter("lookback_years", 4))
        self._lookback = timedelta(days=lookback_years * 365)
        # Schedule monthly rebalancing
        self.schedule.on(
            self.date_rules.month_start(self._equities[0]),
            self.time_rules.after_market_open(self._equities[0], 1),
            self._rebalance
        )
        self.set_warm_up(timedelta(35))
    
    def _rebalance(self):
        if self.is_warming_up:
            return
        # Get historical factor data
        factors = self.history(
            self._factors,
            self._lookback,
            Resolution.DAILY
        )["value"].unstack(0).dropna()
        if factors.empty:
            return
        
        # Calculate 21-day forward returns as labels
        equity_history = self.history(
            self._equities,
            self._lookback,
            Resolution.DAILY,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )
        equity_prices = equity_history["close"].unstack(0) if not equity_history.empty else pd.DataFrame()
        bitcoin_history = self.history(
            [self._bitcoin],
            self._lookback,
            Resolution.DAILY,
            data_normalization_mode=DataNormalizationMode.RAW
        )
        bitcoin_prices = bitcoin_history["close"].unstack(0) if not bitcoin_history.empty else pd.DataFrame()
        prices = pd.concat([equity_prices, bitcoin_prices], axis=1).dropna(how="all")
        labels = prices.pct_change(21).shift(-21).dropna(how="all")
        
        # Train model and make predictions
        prediction_by_symbol = pd.Series(dtype=float)
        for symbol in self._symbols:
            if symbol not in labels.columns:
                continue
            asset_labels = labels[symbol].dropna()
            idx = factors.index.intersection(asset_labels.index)            
            # Fit model
            if len(idx) < 30:
                continue
            self._model.fit(self._scaler.fit_transform(factors.loc[idx]), asset_labels.loc[idx])
            # Predict
            prediction = self._model.predict(self._scaler.transform(factors.iloc[[-1]]))[0]
            if prediction > 0:
                prediction_by_symbol.loc[symbol] = prediction
        if prediction_by_symbol.empty:
            self.liquidate()
            return
        # Calculate weights
        weight_by_symbol = (
            self._gross_exposure * prediction_by_symbol / prediction_by_symbol.sum()
        )
        # Cap Bitcoin weight
        if self._bitcoin in weight_by_symbol.index:
            btc_weight = weight_by_symbol.loc[self._bitcoin]
            if btc_weight > self._max_bitcoin_weight:
                weight_by_symbol.loc[self._bitcoin] = self._max_bitcoin_weight
                remaining = self._gross_exposure - self._max_bitcoin_weight
                non_btc = weight_by_symbol.drop(self._bitcoin)
                if remaining > 0 and not non_btc.empty:
                    weight_by_symbol.loc[non_btc.index] = (
                        remaining * non_btc / non_btc.sum()
                    )
        # Execute trades
        targets = [
            PortfolioTarget(symbol, weight)
            for symbol, weight in weight_by_symbol.items()
        ]
        self.set_holdings(targets, True)
    
    def on_warmup_finished(self):
        self._rebalance()

# More work to be done, still doing some calculations that I can later implement.