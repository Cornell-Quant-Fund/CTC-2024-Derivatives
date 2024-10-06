import numpy as np
import pandas as pd
import yfinance as yf
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Backtester:

  def __init__(self, start_date, end_date, strategy) -> None:
    self.capital : float = 100_000_000
    self.portfolio_value : float = 0

    self.start_date : datetime = start_date
    self.end_date : datetime = end_date
  
    self.user_strategy = strategy
    self.orders : pd.DataFrame = self.user_strategy.generate_orders()
    self.orders["day"] = self.orders["datetime"].apply(lambda x: x.split("T")[0])
    self.orders["expiration_date"] = self.orders["option_symbol"].apply(lambda x: self.get_expiration_date(x))
    self.orders["sort_by"] = pd.to_datetime(self.orders["datetime"])
    self.orders = self.orders.sort_values(by="sort_by")
    self.orders.to_csv("data/example_orders.csv", index=False)

    self.options : pd.DataFrame = pd.read_csv("data/cleaned_options_data.csv")
    self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])
    self.underlying = pd.read_csv("data/underlying_data_hour.csv")
    self.underlying.columns = self.underlying.columns.str.lower()

    self.pnl : List = []
    self.max_drawdown : float = float("-inf")
    self.overall_return : float = 0
    self.sharpe_ratio : float = 0
    self.overall_score : float = 0
    self.open_orders : pd.DataFrame = pd.DataFrame(columns=["day", "datetime", "option_symbol", "action", "order_size", "expiration_date"])

  def get_expiration_date(self, symbol) -> str:
    numbers : str = symbol.split(" ")[3]
    date : str = numbers[:6]
    date_yymmdd : str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
    return date_yymmdd

  def parse_option_symbol(self, symbol) -> List:
    """
    example: SPX   240419C00800000
    """
    numbers : str = symbol.split(" ")[3]
    date : str = numbers[:6]
    date_yymmdd : str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
    action : str = numbers[6]
    strike_price : float = float(numbers[7:]) / 1000
    return [datetime.strptime(date_yymmdd, "%Y-%m-%d"), action, strike_price]

  def calculate_pnl(self):
    delta: timedelta = timedelta(days=1)
    current_date: datetime = self.start_date

    while current_date <= self.end_date:
      for _, row in self.orders.iterrows():
        if str(current_date).split(" ")[0] == str(row["day"]):
          option_metadata: List = self.parse_option_symbol(row["option_symbol"])
          order_size: float = float(row["order_size"])
          strike_price: float = option_metadata[2]

          matching_row = self.options[(self.options["symbol"] == row["option_symbol"]) & 
                                      (self.options["ts_recv"] == row["datetime"])]

          if not matching_row.empty:
            matching_row = matching_row.iloc[0]
          else:
            continue

          ask_price = float(matching_row["ask_px_00"])
          buy_price = float(matching_row["bid_px_00"])
          ask_size = float(matching_row["ask_sz_00"])
          buy_size = float(matching_row["bid_sz_00"])

          if order_size < 0:
            raise ValueError("Order size must be positive")

          if (row["action"] == "B" and order_size > ask_size) or (row["action"] == "S" and order_size > buy_size):
            raise ValueError(f"Order size exceeds available size; order size: {order_size}, ask size: {ask_size}, buy size: {buy_size}")

          if row["action"] == "B":
            options_cost: float = order_size * ask_price + 0.1 * strike_price
            if self.capital >= options_cost:
              self.capital -= options_cost
              self.portfolio_value += options_cost
              self.open_orders.loc[len(self.open_orders)] = row
          else:
            underlying_price: float = float(self.underlying[self.underlying["date"] == row["day"]]["adj close"].iloc[0])
            sold_stock_cost: float = order_size * 100 * underlying_price
            if (self.capital + order_size * buy_price + 0.1 * strike_price) > sold_stock_cost:
              self.capital += order_size * buy_price + 0.1 * strike_price
              self.capital -= sold_stock_cost
              self.portfolio_value += order_size * 100 * underlying_price
              self.open_orders.loc[len(self.open_orders)] = row

      for _, order in self.open_orders.iterrows():
        option_metadata: List = self.parse_option_symbol(order["option_symbol"])
        if str(order["expiration_date"]) == str(current_date).split(" ")[0]:
          underlying_price: float = float(self.underlying[self.underlying["date"] == order["day"]]["adj close"].iloc[0])
          put_call: str = option_metadata[1]
          strike_price: float = option_metadata[2]
          order_size: float = float(order["order_size"])
          underlying_cost: float = strike_price * 100 * order_size

          if order["action"] == "B":
            if put_call == "C":
              if underlying_price > strike_price:
                profit = 100 * order_size * (underlying_price - strike_price)
                self.capital += profit
                self.portfolio_value -= underlying_cost
            else:
              if underlying_price < strike_price:
                self.capital += underlying_cost
          else:
            if put_call == "C":
              if underlying_price > strike_price:
                loss = order_size * 100 * (underlying_price - strike_price)
                self.portfolio_value -= loss
              else:
                if underlying_price < strike_price:
                  cost = order_size * 100 * (strike_price - underlying_price)
                  self.capital -= cost
                  self.portfolio_value += cost
          
          self.capital = max(self.capital, 0)
          self.portfolio_value = max(self.portfolio_value, 0)

      current_date += delta
      print(str(current_date), "capital:", self.capital, "portfolio value:", self.portfolio_value, "total pnl:", (self.capital + self.portfolio_value), "open orders:", len(self.open_orders))

      self.pnl.append(self.capital + self.portfolio_value)

  def compute_overall_score(self):
    self.max_drawdown = (max(self.pnl) - min(self.pnl)) / max(self.pnl)
    print(f"Max Drawdown: {self.max_drawdown}")

    self.overall_return = 100 * ((self.pnl[-1] - 1_000_000) / 1_000_000)
    print(f"Overall Return: {self.overall_return}%")

    percentage_returns = []
    prev = 100_000_000
    for i in range(len(self.pnl)):
      percentage_returns.append(self.pnl[i] / prev)
      prev = self.pnl[i]

    avg_return = np.mean(percentage_returns)
    std_return = np.std(percentage_returns)
    
    if std_return > 0.0:
      risk_free_rate = 0.03 / 252
      self.sharpe_ratio = (avg_return - risk_free_rate) / std_return
      print(f"Sharpe Ratio: {self.sharpe_ratio}")
    else:
      self.sharpe_ratio = 0.0
      print("Sharpe Ratio: Undefined (Standard Deviation = 0)")

    if self.max_drawdown > 0 and self.sharpe_ratio > 0:
      self.overall_score = (self.overall_return / self.max_drawdown) * self.sharpe_ratio
      print(f"Overall Score: {self.overall_score}")
    else:
      print("Cannot calculate overall score (Max Drawdown or Sharpe Ratio <= 0)")

  def plot_pnl(self):
    if not isinstance(self.pnl, list) or len(self.pnl) == 0:
      print("PNL data is not available or empty")

    plt.figure(figsize=(10, 6))
    plt.plot(self.pnl, label="PnL", color="blue", marker="o", linestyle="-", markersize=5)
    
    plt.title("Profit and Loss Over Time", fontsize=14)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("PnL (Profit/Loss)", fontsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()